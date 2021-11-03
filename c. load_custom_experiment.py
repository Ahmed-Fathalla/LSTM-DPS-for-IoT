'''
- "utils._experiment.load_experiment" presents experiment phases' results and the all output of the experiment.
- Select an Experiment form "Experiments output" directory to report its results:
    - for example, to ckeck "Exp Intel-DS Mote_1 Epsilon_0.4", where Intel-DS dataset is used for Mote_id = 1 and error_threshols(epsilon)=0.4, make a call to "load_experiment" function with the following parameters, load_experiment( exp_id = 'Experiments/Exp Intel-DS Mote_1 Epsilon_0.4.pkl', summary=True).
    
    The output will be as follows:
    =====================================
        No_of_phases: 3 "exp/2019-12-10 06_29_17PM.pkl" 
            Phase_0  *** title: "initializing"
                 starting_index :0
                 phase_model_architecture : ---
                 model_h5_str : ---
                 model_fit_str : ---
                 build_time :0
                 No_of_observations :4
                 miss_pred_count :4
                 squared_error :0
                 absolute_error :0
                 mse :0.0
                 mae :0.0
                 accumulated_err_ratio : [1.00000, 1.00000, 1.00000, 1.00000]
                 block_err :[]

            Phase_1  *** title: "LMS"
                 starting_index :4
                 phase_model_architecture : ---
                 model_h5_str : ---
                 model_fit_str : ---
                 build_time :0
                 No_of_observations :20
                 miss_pred_count :0
                 squared_error :0.03177884685402371
                 absolute_error :0.6546835845351282
                 mse :0.0015889423427011857
                 mae :0.03273417922675641
                 accumulated_err_ratio : [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
                 block_err :[]

            Phase_2  *** title: "LSTM_starting"
                 starting_index :25
                 phase_model_architecture :                         
                                 --------------------------
                                 LSTM:16
                                 denes_1:8 Activation:relu
                                 denes_2:0 Activation:relu
                                 Dropout:0.1  batch_size:8
                                 --------------------------
                 model_h5_str :mod/2019-12-10 06_33_57PM1575974037.036447.h5
                 model_fit_str :0.0318_0.0134_update_0.0077_0.0111
                 build_time :318.9327299594879
                 No_of_observations :8313
                 miss_pred_count :93
                 squared_error :339.16191163456926
                 absolute_error :1415.8205796473676
                 mse :0.04079897890467572
                 mae :0.17031403580504842
                 accumulated_err_ratio : [0.01120, 0.01119, 0.01119, 0.01119, 0.01119, 0.01119, 0.01119]
                 block_err :[[0, 0], [0, 0], [0, 0], [6, 0], [7, 0], [2, 0], [1, 0], [4, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0], [3, 0], [7, 0], [9, 0], [1, 0], [3, 0], [3, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [5, 0], [10, 1], [2, 0], [2, 0], [1, 0], [3, 0], [0, 0], [0, 0], [1, 0], [0, 0], [3, 0], [2, 0], [2, 0], [7, 0], [4, 0], [2, 0], [0, 0]]

        total_miss_pred_count: 97
        MAE: 0.16992
        MSE: 0.04069
        Experimental results : W_size:3 /3  mote_ID:1   Reduc:98.83%  fit_0.0318_0.0134_update_0.0077_0.0111  time:5.453 phases:3
        Experiment setup     : Diff:True  Hist:True  G_pr:20  Updt:False-200_10_3  Dsze:None  emax:0.4  wind:3 /3   rati:0.7  rnft:2  hptr:5 
        =====================================
    
    
'''
from utils._experiment import load_experiment
exp = load_experiment( exp_id = 'Experiments/Exp Intel-DS Mote_1 Epsilon_0.4.pkl', summary=True)