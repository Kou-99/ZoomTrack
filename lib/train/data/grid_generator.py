import torch
import numpy as np
import torch.nn.functional as F
from cvxopt import matrix, solvers
from scipy.linalg import block_diag

def unwarp_bboxes(bboxes, grid, output_shape):
    """Unwarps a tensor of bboxes of shape (n, 4) or (n, 5) according to the grid \
    of shape (h, w, 2) used to warp the corresponding image and the \
    output_shape (H, W, ...)."""
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(2, 0, 1).unsqueeze(0)

    warped_height, warped_width = grid.shape[0:2]
    xgrid = 2 * (bboxes[:, 0:4:2] / warped_width) - 1
    ygrid = 2 * (bboxes[:, 1:4:2] / warped_height) - 1
    grid = torch.stack((xgrid, ygrid), dim=2).unsqueeze(0)

    # warped_bboxes has shape (2, num_bboxes, 2)
    warped_bboxes = F.grid_sample(
        img, grid, align_corners=True, padding_mode="border").squeeze(0)
    bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * output_shape[1]
    bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * output_shape[0]

    return bboxes

def unwarp_bboxes_batch(bboxes, grid):
    """
    args:
        bboxes: torch.Tensor (bs,n,4) xyxy [0,1]
        grid: torch.Tensor (bs, h, w,2)
        output_shape: tuple h,w
    """
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(0, 3, 1, 2)

    xgrid = 2 * (bboxes[:, :, 0:4:2]) - 1
    ygrid = 2 * (bboxes[:, :, 1:4:2]) - 1
    grid = torch.stack((xgrid, ygrid), dim=3)

    warped_bboxes = F.grid_sample(
        img, grid, align_corners=True, padding_mode="border")

    bboxes[:, :, 0:4:2] = (warped_bboxes[:, 0,...] + 1) / 2 
    bboxes[:, :, 1:4:2] = (warped_bboxes[:, 1,...] + 1) / 2 

    return bboxes

def warp_boxes(bboxes, grid):
    """
    args:
        bboxes: torch.Tensor (bs,n,4) xyxy [0,1]
        grid: torch.Tensor (bs, h, w,2) [-1,1] xy
    returns:
        warpped_boxes: torch.Tensor (bs,n,4) xyxy [0,1]
    """
    bs, h, w, _ = grid.shape
    whwh = torch.tensor([w,h,w,h], device=grid.device)
    _, n, _ = bboxes.shape
    bboxes_grid_scale = bboxes.clone() * 2 - 1
    points = torch.cat([bboxes_grid_scale[...,:2], bboxes_grid_scale[...,2:]], dim=1) # (bs, 2n, 2)
    x_idx = torch.searchsorted(grid[...,0].contiguous(), points[:,None,:,0].expand(-1,h,-1).contiguous()) #(bs,h,2n)
    xs = torch.cat([grid[...,1], grid[:,:,-2:-1,1]], dim=-1).gather(dim=-1, index=x_idx) #(bs,h,2n)
    y_idx = torch.searchsorted(xs.permute(0,2,1).contiguous(), points[:,:,1:].contiguous()) #(bs, 2n, 1)
    x_idx_final = torch.cat([x_idx, x_idx[:,-1:,:]], dim=-2).gather(dim=1, index=y_idx.permute(0,2,1)).permute(0,2,1) #(bs, 2n, 1)
    y_idx_br = torch.where(y_idx>=h, h-1, y_idx) #(bs, 2n, 1)
    x_idx_br = torch.where(x_idx_final>=w, w-1, x_idx_final) #(bs, 2n, 1)
    y_idx_tl = torch.where(y_idx_br-1<=-1, 1, y_idx_br-1)
    x_idx_tl = torch.where(x_idx_br-1<=-1, 1, x_idx_br-1)
    idx_br = torch.cat([x_idx_br, y_idx_br], dim=-1) #(bs,2n,2) xy
    idx_tl = torch.cat([x_idx_tl, y_idx_tl], dim=-1) #(bs,2n,2) xy
    grid_point_idx = torch.stack([idx_tl, idx_br], dim=2) #(bs,2n,top2, 2) xy
    idx = grid_point_idx[...,1] * w + grid_point_idx[...,0] # (bs, 2n, top2)
    
    flatten_grid = grid.flatten(1,2).unsqueeze(1) #(bs, 1, hw, 2)
    grid_point = flatten_grid.expand(-1,2*n,-1,-1).gather(dim=-2, index=idx.unsqueeze(-1).expand(-1,-1,-1,2)) #(bs, 2n, topk, 2)
    p1 = grid_point[:,:,0,:]
    p2 = grid_point[:,:,1,:]
    p = points.squeeze(2)
    w1 = torch.div(p2-p, p2-p1)
    w2 = torch.div(p-p1, p2-p1)
    target = w1 * grid_point_idx[:,:,0,:] + w2 * grid_point_idx[:,:,1,:]
    out_box = torch.cat(target.split(n, dim=1), dim=-1)
    out_box_norm = out_box / whwh[None,None,:]
    return out_box_norm

class QPGrid():
    def __init__(self, 
                 grid_shape=(31, 51),
                 bandwidth_scale=1,
                 amplitude_scale=64,
                 grid_type='qp',
                 zoom_factor=1.5,
                 loss_dict={'asap':1},
                 constrain_type = 'none',
                 solver='cvxopt'
                 ):
        super().__init__()
        self.grid_shape = grid_shape
        self.bandwidth_scale = bandwidth_scale
        self.amplitude_scale = amplitude_scale
        self.grid_type = grid_type
        self.zoom_factor = zoom_factor
        self.loss_dict = loss_dict
        self.constrain_type = constrain_type
        self.solver = solver
        
        self.precompute_ind()

    def precompute_ind(self):
        m, n = self.grid_shape
        m -= 1
        n -= 1
        self.asap_k_row = np.concatenate([np.arange(m*n, dtype=int), np.arange(m*n, dtype=int)])
        self.asap_k_col = np.concatenate([np.arange(m, dtype=int).repeat(n), np.tile((np.arange(m, m+n, dtype=int)), m)])
        
        self.ts_r_row = np.arange(2*m*n)
        self.ts_r_col = np.concatenate([np.arange(m).repeat(n), np.tile((np.arange(n)+m), m)])
        
        self.ts_v_row = np.arange(2*m*n, dtype=int)
        self.ts_v_col = np.zeros(2*m*n, dtype=int)
        
        
        self.naive_constrain_A_np = np.stack([np.concatenate([np.ones(m), np.zeros(n)]), np.concatenate([np.zeros(m), np.ones(n)])])

    def bbox2sal(self, batch_bboxes, img_metas, jitter=None):
        """
        taken from https://github.com/tchittesh/fovea
        """
        h_out, w_out = self.grid_shape
        sals = []
        for i in range(len(img_metas)):
            h, w, _ = img_metas[i]['pad_shape']
            bboxes = batch_bboxes[i]
            if len(bboxes) == 0:  # zero detections case
                sal = np.ones((h_out, w_out)).expand_dims(0)
                sal /= sal.sum()
                sals.append(sal)
                continue
            
            if isinstance(batch_bboxes, torch.Tensor):
                if batch_bboxes.is_cuda:
                    bboxes = bboxes.cpu()
                bboxes = bboxes.numpy()
            cxy = bboxes[:, :2] + 0.5*bboxes[:, 2:]
            if jitter is not None:
                cxy += 2*jitter*(np.random.randn(*cxy.shape)-0.5)
            widths = (bboxes[:, 2] * self.bandwidth_scale).reshape(-1, 1)
            heights = (bboxes[:, 3] * self.bandwidth_scale).reshape(-1, 1)

            X, Y = np.meshgrid(
                np.linspace(0, w, w_out, dtype=np.float32),
                np.linspace(0, h, h_out, dtype=np.float32),
                indexing='ij'
            )
            grids = np.stack((X.flatten(), Y.flatten()), axis=1).T

            m, n = cxy.shape[0], grids.shape[1]

            norm1 = np.tile((cxy[:, 0:1]**2/widths + cxy[:, 1:2]**2/heights), (m, n))
            norm2 = grids[0:1, :]**2/widths + grids[1:2, :]**2/heights
            norms = norm1 + norm2

            cxy_norm = cxy
            cxy_norm[:, 0:1] /= widths
            cxy_norm[:, 1:2] /= heights

            distances = norms - 2*cxy_norm.dot(grids)

            sal = np.exp((-0.5 * distances))
            sal = self.amplitude_scale * (sal / (0.00001+sal.sum(axis=1, keepdims=True)))  # noqa: E501, normalize each distribution
            sal += 1/(self.grid_shape[0]*self.grid_shape[1])
            sal = sal.sum(axis=0)
            sal /= sal.sum()
            
            # scale saliency to peak==1
            sal = 1 / sal.max() * sal
            
            sal = sal.reshape(w_out, h_out).T[np.newaxis, ...]  # noqa: E501, add channel dimension
            sals.append(sal)
        return np.stack(sals)
    
    def asap_loss(self, saliency, im_shape):
        m, n = saliency.shape
        h, w = im_shape
        mh = m/h
        nw = n/w
        k_left = (block_diag(*np.split(saliency, m ,axis=0)) * mh).T
        k_right = np.concatenate([-np.diag(saliency[i,:])*nw for i in range(m)], axis=0)
        k = np.concatenate([k_left, k_right], axis=1)
        P = 2 * np.matmul(k.T, k)
        q = np.zeros((m+n,1))
        P_mat = P
        q_mat = q
        return P_mat, q_mat
    
    def asap_ts(self, saliency, im_shape):
        m, n = saliency.shape
        h, w = im_shape
        saliency_sq = np.square(saliency)
        sal_x = saliency_sq.sum(axis=1)
        sal_y = saliency_sq.sum(axis=0)
        p_tl = np.diag(sal_x*(2*m*m/h/h+2))
        p_br = np.diag(sal_y*(2*n*n/w/w+2))
        p_tr = saliency_sq*(-2*m*n/h/w)
        P = np.block([
            [p_tl, p_tr],
            [p_tr.T, p_br]
        ])
        q = np.concatenate([sal_x*(-2*h/m/self.zoom_factor), sal_y*(-2*w/n/self.zoom_factor)])[:,None]
        return P, q
    
    def asap_ts_weight(self, saliency, im_shape, loss_list):
        """
        loss_list: list [ts_weight, asap_weight]
        """
        m, n = saliency.shape
        h, w = im_shape
        w_ts, w_asap = loss_list
        saliency_sq = np.square(saliency)
        sal_x = saliency_sq.sum(axis=1)
        sal_y = saliency_sq.sum(axis=0)
        p_tl = np.diag(sal_x*(2*w_asap*m*m/h/h+2*w_ts))
        p_br = np.diag(sal_y*(2*w_asap*n*n/w/w+2*w_ts))
        p_tr = saliency_sq*(-2*m*n/h/w*w_asap)
        P = np.block([
            [p_tl, p_tr],
            [p_tr.T, p_br]
        ])
        q = np.concatenate([sal_x*(-2*h/m/self.zoom_factor*w_ts), sal_y*(-2*w/n/self.zoom_factor*w_ts)])[:,None]
        return P, q    
    
    
    def ts_loss(self, saliency, im_shape):
        m, n = saliency.shape
        h, w = im_shape
        ideal_h =  h / m / self.zoom_factor
        ideal_w =  w / n / self.zoom_factor
        r_top = np.concatenate([(block_diag(*np.split(saliency, m ,axis=0)) ).T, np.zeros((m*n, n))], axis=1) 
        r_bottom = np.concatenate([np.zeros((m*n, m)), np.concatenate([np.diag(saliency[i,:]) for i in range(m)], axis=0)], axis=1) 
        v = np.concatenate([saliency[i,:] for i in range(m)])[:,None]
        R = np.concatenate([r_top, r_bottom], axis=0)
        V = np.concatenate([v*ideal_h, v*ideal_w], axis=0)
        P = 2 * np.matmul(R.T, R)
        q = -2 * np.matmul(R.T, V)
        P_mat = P
        q_mat = q
        return P_mat, q_mat
    
    
    def naive_constarin(self, saliency, im_shape):
        h, w = im_shape
        
        A = self.naive_constrain_A_np
        b = np.array([h, w])
        
        return A, b, None, None
    
    def get_qp_loss(self, saliency, im_shape):
        """
        args:
            saliency: np.ndarray (m,n)
            im_shape: shape of original image (h,w)
        returns:
            P, q, A, b, G, h
            min 1/2 * x^TPx + q^Tx
            s.t. Gx <= h
                 Ax = b
        """
        loss_func = {
            'asap': self.asap_loss,
            'ts': self.ts_loss,
            'asap_ts': self.asap_ts,
            'asap_ts_weight': self.asap_ts_weight
        }
        constrain_func = {
            'none': self.naive_constarin,
        }
        P_mat = 0
        q_mat = 0
        for loss_name, weight in self.loss_dict.items():
            if loss_name == 'asap_ts_weight':
                P_mat_single, q_mat_single = loss_func[loss_name](saliency, im_shape, weight)
                P_mat += P_mat_single
                q_mat += q_mat_single
            else:
                P_mat_single, q_mat_single = loss_func[loss_name](saliency, im_shape)
                P_mat += weight * P_mat_single
                q_mat += weight * q_mat_single
        A_mat, b_mat, G_mat, h_mat = constrain_func[self.constrain_type](saliency, im_shape)
        return P_mat, q_mat, A_mat, b_mat, G_mat, h_mat

    def cvxopt(self, P, q, A, b, G, h):
        P_mat = matrix(np.float64(P))
        q_mat = matrix(np.float64(q))
        A_mat = matrix(np.float64(A))
        b_mat = matrix(np.float64(b))
        sol = solvers.qp(P=P_mat,q=q_mat,A=A_mat,b=b_mat)
        x_array = np.array(sol['x'])
        return torch.tensor(x_array,dtype=torch.float32)

    def solve(self, P, q, A, b, G, h):
        """
        args:
            all input should be np.ndarry, np.float32
        returns:
            torch.Tensor torch.float32
        """
        solver_fn = {
            'cvxopt': self.cvxopt,
        }
        return solver_fn[self.solver](P, q, A, b, G, h)
    
    def qp_grid(self, img, saliency, out_shape, **kwargs):
        """
        img: (bs, channel, h, w)
        saliency: (bs, h, w)
        """
        bs, m, n = saliency.shape
        m -= 1
        n -= 1
        grid_list = []
        for i, sal in enumerate(saliency):
            sal_center = (sal[:-1,:-1] + sal[1:,1:] + sal[:-1,1:] + sal[1:,:-1]) / 4
            h, w = img[i].shape[1:]
            P_mat, q_mat, A_mat, b_mat, G_mat, h_mat = self.get_qp_loss(sal_center, img[i, 0].shape)
            x = self.solve(P_mat, q_mat, A_mat, b_mat, G_mat, h_mat)


            ygrid = torch.cat([torch.zeros(1,1),torch.cumsum(x[:m], 0)], dim=0) / h
            ygrid = torch.clamp(ygrid*2-1, min=-1, max=1)
            ygrid = ygrid.view(-1, 1, self.grid_shape[0], 1)
            ygrid = ygrid.expand(-1, 1, *self.grid_shape)

            xgrid = torch.cat([torch.zeros(1,1),torch.cumsum(x[-n:], 0)], dim=0) / w
            xgrid = torch.clamp(xgrid*2-1, min=-1, max=1)
            xgrid = xgrid.view(-1, 1, 1, self.grid_shape[1])
            xgrid = xgrid.expand(-1, 1, *self.grid_shape)
            
            grid = torch.cat((xgrid, ygrid), 1)
            grid_list.append(grid)
            
        
        grids = torch.cat(grid_list, dim=0)
        grids = F.interpolate(grids, size=out_shape, mode='bilinear',
                             align_corners=True)  
        return grids.permute(0, 2, 3, 1)

    def gen_grid_from_saliency(self, img, saliency, out_shape, **kwargs):
        grid_func = {
            'qp': self.qp_grid
        }
        return grid_func[self.grid_type](img, saliency, out_shape, **kwargs)

    def forward(self, imgs, img_metas, gt_bboxes, out_shape, jitter=None):
        """
        args:
            imgs: torch.Tensor (bs, channel, h, w)
            img_metas: list of dict, len==bs, dict containing:
                        pad_shape: tuple, (h, w, c)
            gt_bboxes: torch.Tensor (bs, num_box, 4), dtype: float32
        returns:

        """
        if isinstance(gt_bboxes, torch.Tensor):
            batch_bboxes = gt_bboxes
        else:
            if len(gt_bboxes[0].shape) == 3:
                batch_bboxes = gt_bboxes[0].clone()  # noqa: E501, removing the augmentation dimension
            else:
                batch_bboxes = [bboxes.clone() for bboxes in gt_bboxes]
        device = batch_bboxes[0].device
        saliency = self.bbox2sal(batch_bboxes, img_metas)
        grid = self.gen_grid_from_saliency(imgs, np.squeeze(saliency, axis=1), out_shape)
        
        return grid.to(device), saliency
    
def build_data_search_grid_generator(cfg):
    loss_dict = dict()
    for loss_name, loss_weight in zip(cfg.DATA.SEARCH.GRID.GENERATOR.LOSS.NAMES, cfg.DATA.SEARCH.GRID.GENERATOR.LOSS.WEIGHTS):
        loss_dict[loss_name] = loss_weight
    grid_generator = QPGrid(
        grid_shape=cfg.DATA.SEARCH.GRID.SHAPE,
        bandwidth_scale=cfg.DATA.SEARCH.GRID.GENERATOR.BANDWIDTH_SCALE,
        amplitude_scale=1,
        zoom_factor=cfg.DATA.SEARCH.GRID.GENERATOR.ZOOM_FACTOR,
        grid_type=cfg.DATA.SEARCH.GRID.TYPE,
        loss_dict = loss_dict
    )
    return grid_generator