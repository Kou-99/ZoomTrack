from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/ytkou/ZoomTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/ytkou/ZoomTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/ytkou/ZoomTrack/data/itb'
    settings.lasot_extension_subset_path = '/home/ytkou/ZoomTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/ytkou/ZoomTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/ytkou/ZoomTrack/data/lasot'
    settings.network_path = '/home/ytkou/ZoomTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/ytkou/ZoomTrack/data/nfs'
    settings.otb_path = '/home/ytkou/ZoomTrack/data/otb'
    settings.prj_dir = '/home/ytkou/ZoomTrack'
    settings.result_plot_path = '/home/ytkou/ZoomTrack/output/test/result_plots'
    settings.results_path = '/home/ytkou/ZoomTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/ytkou/ZoomTrack/output'
    settings.segmentation_path = '/home/ytkou/ZoomTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/ytkou/ZoomTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/ytkou/ZoomTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/ytkou/ZoomTrack/data/trackingnet'
    settings.uav_path = '/home/ytkou/ZoomTrack/data/uav'
    settings.vot18_path = '/home/ytkou/ZoomTrack/data/vot2018'
    settings.vot22_path = '/home/ytkou/ZoomTrack/data/vot2022'
    settings.vot_path = '/home/ytkou/ZoomTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

