'''
To re-run an experiment and get the same results reported in the paper (using the same model we got during our experiment):
    - "utils._experiment.re_run_experiment" make a run of the experiments using the same LSTM-DPS model we got during the experiment phases.
    - Select an Experiment form "Experiments output" directory to rerun the experiment and get the same score reported in the paper:
        - for example, to rerun 'Experiments/Exp Benzene-DS Epsilon_0.002.pkl', where Benzene-DS dataset is used for error_threshols(epsilon)=0.002, make a call to "re_run_experiment" function with the following parameters, re_run_experiment( exp_id = 'Experiments/Exp Benzene-DS Epsilon_0.002.pkl' ).
        
        The output will be as follows:

            Loading model: models/LSTM_DPS_model Benzene-DS Epsilon_0.002 Phase_2.h5 ***********
            Loading model: models/LSTM_DPS_model Benzene-DS Epsilon_0.002 Phase_3.h5 ***********
            Loading model: models/LSTM_DPS_model Benzene-DS Epsilon_0.002 Phase_4.h5 ***********

            =====================================
            No_of_phases: 5
                Phase_0  *** title: "initializing"
                     mse :0.0
                     No_of_observations :4
                     phase_model_architecture : ---
                     accumulated_err_ratio : [1.00000, 1.00000, 1.00000, 1.00000]
                     mae :0.0
                     model_fit_str : ---
                     absolute_error :0
                     build_time :0
                     model_h5_str : ---
                     squared_error :0
                     block_err :[]
                     starting_index :0
                     miss_pred_count :4

                Phase_1  *** title: "LMS"
                     mse :0.6971234464384839
                     No_of_observations :20
                     phase_model_architecture : ---
                     accumulated_err_ratio : [0.42857, 0.40000, 0.43750, 0.47059, 0.50000, 0.52632, 0.50000]
                     mae :0.5134733892144137
                     model_fit_str : ---
                     absolute_error :10.269467784288272
                     build_time :0
                     model_h5_str : ---
                     squared_error :13.942468928769678
                     block_err :[]
                     starting_index :4
                     miss_pred_count :10

                Phase_2  *** title: "LSTM_starting"
                     mse :0.5897270066210515
                     No_of_observations :3000
                     phase_model_architecture :                         
                                     --------------------------
                                     LSTM:1
                                     denes_1:1 Activation:relu
                                     denes_2:0 Activation:
                                     Dropout:0.0  batch_size:0
                                     --------------------------
                     accumulated_err_ratio : [0.49833, 0.49816, 0.49800, 0.49783, 0.49767, 0.49783, 0.49767]
                     mae :0.4610759248939154
                     model_fit_str :Loaded_model
                     absolute_error :1383.2277746817463
                     build_time :0
                     model_h5_str :models/LSTM_DPS_model Benzene-DS Epsilon_0.002 Phase_2.h5
                     squared_error :1769.1810198631545
                     block_err :[[122, 1], [95, 0], [91, 0], [83, 0], [107, 1], [106, 1], [95, 0], [103, 1], [101, 1], [89, 0], [100, 1], [88, 0], [101, 1], [102, 1], [110, 1]]
                     starting_index :25
                     miss_pred_count :1493

                Phase_3  *** title: "Update_3051"
                     mse :0.6566610427117288
                     No_of_observations :1800
                     phase_model_architecture :                         
                                     --------------------------
                                     LSTM:16
                                     denes_1:8 Activation:linear
                                     denes_2:0 Activation:
                                     Dropout:0.2  batch_size:0
                                     --------------------------
                     accumulated_err_ratio : [0.43032, 0.43008, 0.42984, 0.43016, 0.43048, 0.43079, 0.43056]
                     mae :0.5148907302317941
                     model_fit_str : ---
                     absolute_error :926.8033144172294
                     build_time :0
                     model_h5_str :models/LSTM_DPS_model Benzene-DS Epsilon_0.002 Phase_3.h5
                     squared_error :1181.9898768811117
                     block_err :[[88, 0], [83, 0], [68, 0], [54, 0], [61, 0], [91, 0], [106, 1], [103, 1], [122, 1]]
                     starting_index :3051
                     miss_pred_count :776

                Phase_4  *** title: "Update_4851"
                     mse :0.6403178399471222
                     No_of_observations :4167
                     phase_model_architecture :                         
                                     --------------------------
                                     LSTM:8
                                     denes_1:4 Activation:relu
                                     denes_2:4 Activation:linear
                                     Dropout:0.0  batch_size:0
                                     --------------------------
                     accumulated_err_ratio : [0.44052, 0.44065, 0.44079, 0.44092, 0.44082, 0.44071, 0.44084]
                     mae :0.5023569390640819
                     model_fit_str : ---
                     absolute_error :2093.321365080029
                     build_time :0
                     model_h5_str :models/LSTM_DPS_model Benzene-DS Epsilon_0.002 Phase_4.h5
                     squared_error :2668.2044390596584
                     block_err :[[116, 1], [96, 0], [122, 1], [109, 1], [83, 0], [85, 0], [110, 1], [95, 0], [91, 0], [92, 0], [72, 0], [92, 0], [95, 0], [51, 0], [75, 0], [81, 0], [62, 0], [86, 0], [90, 0], [81, 0]]
                     starting_index :4851
                     miss_pred_count :1838

            total_miss_pred_count: 4121
            MAE: 0.49095
            MSE: 0.62662
            Experimental results : W_size:3 /3  mote_ID:Benzene  Reduc:54.16%  fit_Loaded_model  time:0.221 phases:5
            Experimental setup   : Diff:True  Hist:False  G_pr:23  Updt:True-200_100_3  Dsze:800  emax:2  wind:3 /3   rati:0.7  rnft:2  hptr:5 
            =====================================
'''


from utils._experiment import re_run_experiment
_ = re_run_experiment( exp_id = 'Experiments/Exp Benzene-DS Epsilon_0.002.pkl' )