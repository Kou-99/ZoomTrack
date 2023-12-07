import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.train.data.grid_generator import unwarp_bboxes, unwarp_bboxes_batch, QPGrid
from lib.utils.box_ops import box_cxcywh_to_xywh


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        if self.cfg.TEST.SEARCH.USE_GRID:
            loss_dict = dict()
            for loss_name, loss_weight in zip(self.cfg.TEST.SEARCH.GRID.GENERATOR.LOSS.NAMES, self.cfg.TEST.SEARCH.GRID.GENERATOR.LOSS.WEIGHTS):
                loss_dict[loss_name] = loss_weight
            self.search_grid_generator = QPGrid(
                amplitude_scale=1,
                bandwidth_scale=self.cfg.TEST.SEARCH.GRID.GENERATOR.BANDWIDTH_SCALE,
                grid_type='qp',
                zoom_factor=self.cfg.TEST.SEARCH.GRID.GENERATOR.ZOOM_FACTOR,
                loss_dict=loss_dict,
                grid_shape=self.params.cfg.TEST.SEARCH.GRID.SHAPE
            )
            self.out_shape = (self.params.cfg.TEST.SEARCH.SIZE, self.params.cfg.TEST.SEARCH.SIZE)

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        if not self.cfg.TEST.SEARCH.USE_GRID:
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
        else:
            x_patch_arr, x_amask_arr, resize_factor = sample_target(image, self.state, self.params.search_factor)  # (x1, y1, w, h)
            img_shape_ori = x_patch_arr.shape
            img_metas = [
                {'pad_shape': img_shape_ori}
            ]
            img_tensor = torch.tensor(x_patch_arr, dtype=torch.float32).permute(2,0,1)[None,...]
            x_patch_h, x_patch_w, _ = img_shape_ori
            saliency_box = torch.tensor(self.state, dtype=torch.float32)
            saliency_box[0] = x_patch_w / 2 - saliency_box[2] / 2
            saliency_box[1] = x_patch_h / 2 - saliency_box[3] / 2
            grid, saliency = self.search_grid_generator.forward(img_tensor.cuda(), img_metas, saliency_box[None,None,:].cuda(), self.out_shape, jitter=0)
            x_patch_arr = F.grid_sample(img_tensor.cuda(), grid, align_corners=True)
            x_patch_arr = x_patch_arr.cpu()[0].permute(1,2,0).numpy()
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        if not self.cfg.TEST.SEARCH.USE_GRID:
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        else:
            
            pred_box = pred_boxes.mean(dim=0, keepdim=True)
            search_whwh = torch.tensor([self.params.search_size, self.params.search_size, self.params.search_size, self.params.search_size], device=pred_box.device)

            pred_box = torch.cat([pred_box[:,:2]-pred_box[:,2:]/2, pred_box[:,:2]+pred_box[:,2:]/2], dim=-1)
            # search_whwh = torch.tensor([self.params.search_w, self.params.search_h, self.params.search_w, self.params.search_h], device=pred_box.device)
            unwraped_box = unwarp_bboxes(pred_box * search_whwh, grid[0], img_shape_ori)
            unwraped_box = torch.cat([(unwraped_box[:,2:]+unwraped_box[:,:2])/2, unwraped_box[:,2:]-unwraped_box[:,:2]], dim=-1).cpu()
            if self.debug:
                # only display top-k box
                feat_x = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
                feat_y = self.cfg.TEST.SEARCH.SIZE // self.cfg.MODEL.BACKBONE.STRIDE
                feat_xyxy = torch.tensor([feat_x, feat_y, feat_x, feat_y])
                topk_score, topk_index = torch.topk(pred_score_map.squeeze().flatten(), 4)
                topk_index2d = torch.stack([topk_index // pred_score_map.shape[-1], topk_index % pred_score_map.shape[-1]], dim=1)
                topk_pred_boxes_raw = self.cal_box_from_coord(out_dict, topk_index2d.to(out_dict['score_map'].device), resize_factor, H , W, reduce='raw')
                topk_pred_boxes_raw_xyxy = torch.cat([topk_pred_boxes_raw[:,:2]-topk_pred_boxes_raw[:,2:]/2, topk_pred_boxes_raw[:,:2]+topk_pred_boxes_raw[:,2:]/2], dim=-1)
                unwraped_topk_pred_boxes_raw = unwarp_bboxes(topk_pred_boxes_raw_xyxy * search_whwh.cuda(), grid[0], img_shape_ori)
                full_grid = F.interpolate(grid.permute(0,3,1,2), scale_factor=resize_factor)

                # wrong gt_box to make program work
                gt_box = transform_image_to_crop(torch.tensor(info['gt_bbox']), torch.tensor(self.state), resize_factor, x_patch_h)
                pred_boxes_ori = unwraped_topk_pred_boxes_raw.cpu()
                pred_boxes_ori = torch.cat([(pred_boxes_ori[:,2:]+pred_boxes_ori[:,:2])/2, pred_boxes_ori[:,2:]-pred_boxes_ori[:,:2]], dim=-1)
                pred_boxes_ori = torch.stack([torch.tensor(clip_box(self.map_box_back_fullsize(pred_boxes_ori_single.tolist(), x_patch_h), H, W, margin=10))for pred_boxes_ori_single in pred_boxes_ori], dim=0)
            self.state = clip_box(self.map_box_back_fullsize(unwraped_box[0].tolist(), x_patch_h), H, W, margin=10)
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                self.visdom.register((image, self.state, *pred_boxes_ori[~((pred_boxes_ori == torch.tensor(self.state)).all(1))]), 'Tracking', 1, 'Tracking')

                data_dict = dict(
                    image = x_patch_arr,
                    box = box_cxcywh_to_xywh(topk_pred_boxes_raw.to('cpu')),
                    gt_bbox = gt_box
                )
                self.visdom.register(data_dict, 'SearchRegionNoGrid', 1, 'SearchRegionNoGrid')
                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                if self.cfg.TEST.SEARCH.USE_GRID:
                        x_patch_arr_ori = img_tensor[0].permute(1,2,0).numpy()
                        x_patch_arr_resize = cv2.resize(x_patch_arr_ori, (self.params.search_size, self.params.search_size))
                        self.visdom.register(torch.from_numpy(x_patch_arr_resize).permute(2, 0, 1), 'image', 1, 'resized_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)
    
    def map_box_back_fullsize(self, pred_box: list, full_size: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * full_size
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

    def cal_box_from_coord(self, out_dict, coord,  resize_factor, H , W, reduce='mean'):
        """
        coord - (2) or (num_box, 2) dtype:long
        """
        pred_boxes = self.network.box_head.cal_bbox_from_idx(coord, out_dict['size_map'], out_dict['offset_map'])
        
        pred_boxes = pred_boxes.view(-1, 4)
        if reduce == 'mean':
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            return clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        elif reduce == 'none':
            boxes = []            
            pred_boxes = (pred_boxes * self.params.search_size / resize_factor).tolist()
            for box in pred_boxes:
                boxes.append(torch.tensor(clip_box(self.map_box_back(box, resize_factor), H, W, margin=10)))
            return torch.stack(boxes)
        elif reduce == 'raw':
            return pred_boxes
        else:
            raise NotImplementedError


def get_tracker_class():
    return OSTrack
