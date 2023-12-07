import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import sys

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

def main(tracker_name, tracker_config, dataset_name = 'lasot'):
    trackers = []
    
    """stark"""
    # trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
    #                             run_ids=None, display_name='STARK-S50'))
    # trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
    #                             run_ids=None, display_name='STARK-ST50'))
    # trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
    #                             run_ids=None, display_name='STARK-ST101'))
    """TransT"""
    # trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
    #                             run_ids=None, display_name='TransT_N2', result_only=True))
    # trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
    #                             run_ids=None, display_name='TransT_N4', result_only=True))
    """pytracking"""
    # trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
    # trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
    # trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
    # trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
    # trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
    """ostrack"""
    trackers.extend(trackerlist(name=tracker_name, parameter_name=tracker_config, dataset_name=dataset_name,
                                run_ids=None, display_name=tracker_config))
    # trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
    #                             run_ids=None, display_name='OSTrack384'))


    dataset = get_dataset(dataset_name)
    # dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
    # plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
    #              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'), force_evaluation=True)
    # print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))

if __name__ == '__main__':
    dataset_name = 'lasot' if len(sys.argv)==3 else sys.argv[3]
    main(sys.argv[1], sys.argv[2], dataset_name)
