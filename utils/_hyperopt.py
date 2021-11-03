from hyperopt import hp
hyperopt_space = \
    {
    'nb_epochs' : hp.choice('nb_epochs',[10]),
    'lstm_cells' : hp.choice('lstm_cells',[8,16]),
    'd1_nodes' : hp.choice('d1_nodes',[4,8]),
    'd2_nodes' : hp.choice('d2_nodes',[0,4]),
    'batch_size' : hp.choice('batch_size',[4,8,16]),
    'dropout1': hp.choice('dropout1',[0,0.1,0.2]),
    'activation_1': hp.choice('activation_1',['relu', 'linear']),
    'activation_2': hp.choice('activation_2',['relu', 'linear']),
    }