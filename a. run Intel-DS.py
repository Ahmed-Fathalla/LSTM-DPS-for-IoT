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
from utils._utils import read_Intel_data
from utils._experiment_setup import setup_

df = read_Intel_data()

mote_lst = [1, 11, 13, 30, 49]
Epsilon_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
c=0
import traceback
try:
    print('Start--- ', get_timestamp_m() )
    for mote in mote_lst:
        historical_data =  load_mote_data( df, moteID = mote, date='2004-03-05' )
        test_data = load_mote_data( df, moteID = mote, date= ('2004-03-06', '2004-03-09') ).values
        for setup_['e_max_threshold'] in Epsilon_lst:
            run_experiment(setup_ = setup_, exp_variable='e_max_threshold', 
                   test_data=test_data , 
                   historical_data=historical_data,  
                   exp_str=mote, runs=5, file='mote_30.txt', progress = str((c+1)*100/(len(mote_lst)*len(emax_lst))) )
            c+=1

except Exception as exc:
    print('\n**** Err:\n', get_timestamp_m(),'\n',traceback.format_exc())       