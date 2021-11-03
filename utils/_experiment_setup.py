# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: config file
"""

from ._hyperopt import hyperopt_space
setup_ = {
        # Experiment parameters
            'e_max_threshold': 0.5,        # Error threshold it is set in the code
            'Apply_diff': True,
            'train_on_historical_data': True,
            'initialization_period': 3,    # default is the same value of 'LMS_w'
            'LMS_period':20,

            'LMS_w': 3,
            'LSTM_w': 3,
            'data_size': 800,              # Maximum number of training/validation record that will be used to
                                           # train/update the model

        # updating parameter
            'employ_update_policy':True,
            'block_size':200,
            'block_err_threshold': 130,
            'phase_accumulated_block_err': 3,

        # model parameters
            'train_validate_ratio': 0.7,
            'number_random_fits':1,
            'hyperopt_max_trials':1,
            'hyperopt_space': hyperopt_space,
         }