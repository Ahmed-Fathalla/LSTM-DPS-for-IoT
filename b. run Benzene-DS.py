import warnings
warnings.filterwarnings('ignore')
import traceback
import time
import os, sys

from utils._model import Custom_Model, sort_tuble
from utils._experiment import Experiment, load_experiment
from utils._time_utils import get_TimeStamp_str, get_timestamp_m
from utils._write_to_file import write_to_file
from utils._run import run_experiment,run_exp
from utils._utils import read_Benzene_data
from utils._experiment_setup import setup_

test_data = read_Benzene_data()

setup_['train_on_historical_data']= False
setup_['employ_update_policy'] = True
setup_['block_size'] = 200
setup_['block_err_threshold'] = 140
setup_['phase_accumulated_block_err'] = 3 

Epsilon_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

import traceback
try:
    print('starting...',get_timestamp_m())
    for setup_['e_max_threshold'] in Epsilon_lst:
        a = run_experiment(setup_, 'e_max_threshold', test_data, historical_data=None, exp_str='Benzene', load_model=None,
                           show_loss_results=False, get_summary=False, plt=False, upgrade_verbose=1,
                           print_res=0, runs=5, file='Benzen_%f_update.txt'%setup_['e_max_threshold'], write_all=1)
                           
except Exception as exc:
    print('\n**** Err:\n', get_timestamp_m(),'\n',traceback.format_exc())                        