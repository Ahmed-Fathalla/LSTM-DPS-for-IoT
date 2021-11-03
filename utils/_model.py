# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: config file
"""

import sys
import time
import traceback
import pandas as pd
import numpy as np
import tensorflow as tf
import gc
import operator
import warnings
warnings.filterwarnings('ignore')
try:
    import hyperopt
    from hyperopt import fmin, tpe, STATUS_OK, Trials, STATUS_FAIL
except Exception as exc:
    print('\n**** Err A002: hyperopt can not be imported\n',traceback.format_exc())
try:
    import keras
    from keras.models import Model, load_model
    from keras.layers import Input, Dropout, Dense, LSTM
    from keras import backend as K
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
except Exception as exc:
    print('\n**** Err A003: keras can not be imported\n',traceback.format_exc())

from ._time_utils import get_TimeStamp_str
from utils._time_utils import get_timestamp_m


def sort_tuble(tub, item = 2, ascending = True):
    tub = sorted(tub, key=operator.itemgetter(item), reverse=False)
    if ascending:
        return tub[0]
    else:
        return tub[-1]

def train_test_split(x_data, y_data, train_ratio = 0.4):
    train_obs = int(x_data.shape[0]*train_ratio)
    X_train, y_train = x_data[:train_obs], y_data[:train_obs]
    X_valid, y_valid = x_data[train_obs:], y_data[train_obs:]
    return X_train, y_train, X_valid, y_valid

def get_data_look_back(dataseries, window_size = 3, reshape_ = 1, Apply_diff = True, verbose = 0):
    '''
    this function do the following:
        1. takes a dataseries (vector),
        2. makes its differences, and
        3. create number of 'window_size' lag features
    '''
    if not isinstance(dataseries, pd.core.series.Series):
        dataseries = pd.Series(dataseries)
    df = dataseries.to_frame()
    df.columns = ['temperature']
    df.index = range(len(df))
    if Apply_diff:
        df['x_0'] = df['temperature'].diff()#.shift(-1) fat_1
        df = df[1:] # remove the first observation which has a difference of NaN
    else:
        df['x_0'] = df['temperature']
    for i in range(1, window_size):
        df['x_'+str(i)] = df.x_0.shift(-1*i)
    df['y'] = df.x_0.shift(-1*(window_size))

    col_names = ['x_'+str(i)for i in range(0,window_size)]
    df['curr_temp'] = df['temperature'].shift(-1*(window_size)) # (window_size+1) fat_2
    df = df[:-1*(window_size)] # reomve observations of NaN values
    df = df[[*col_names, 'y', 'curr_temp']]

    y_data = df.pop('y')

    if reshape_:
        x_data = np.reshape(df[col_names].values, (df[col_names].shape[0], df[col_names].shape[1], 1))
    else:
        x_data = df[col_names].values

    if verbose:
        print('x_data:', x_data.shape)

    return x_data, y_data

def get_lst_str(lst):
    return ', '.join(['%-.6f'%i for i in lst])

class Custom_Model:
    def __init__(self, setup_dict, historical_data=None):
        self.setup_ = setup_dict
        if historical_data is not None:
            self.prepare_data(historical_data)

        self.model = None
        self.best_architecture_dict = None  # model architecture dictionary
        self.model_h5_str = None
        self.state = None

    def build_fast_model(self):
        if self.best_architecture_dict is None:self.best_architecture_dict={}
        self.best_architecture_dict['nb_epochs'] = 1
        self.best_architecture_dict['lstm_cells'] = 1
        self.best_architecture_dict['d1_nodes'] = 1
        self.best_architecture_dict['activation_1'] = 'relu'
        self.best_architecture_dict['d2_nodes'] = 0
        self.best_architecture_dict['activation_2'] = ''
        self.best_architecture_dict['dropout1'] = 0
        self.best_architecture_dict['batch_size'] = 32
        # model_architecture_str = self.get_model_architecture_str()
        # fit_str = 'fast_model'
        # ep_train_valid_str = None



        return self.get_model_architecture_str()#, self.model_h5_str, fit_str, ep_train_valid_str

    def prepare_data(self, historical_data, Apply_diff = None, get_data_size = False):
        if Apply_diff==None:Apply_diff=self.setup_['Apply_diff']
        if self.setup_['data_size']!=None:
            try:historical_data=historical_data[-self.setup_['data_size']:]
            except Exception as exc:print('****** Err:000E: self.setup_["data_size"]')
        x_data, y_data = get_data_look_back(dataseries = historical_data,
                                        Apply_diff = Apply_diff,
                                        window_size = self.setup_['LSTM_w'])
        self.X_train, self.y_train, self.X_valid, self.y_valid = train_test_split(x_data, y_data, self.setup_['train_validate_ratio'])
        if len(self.X_train)< 1:
            assert False,'Err:006A: prepare_data: Data is not enough to train/fit the model'
        elif len(self.X_valid) < 1:
            assert False,'Err:006B: prepare_data: Data is not enough to validate the model'
        if get_data_size:
            print('X_train.shape ----------' , self.X_train.shape )
            print('X_valid.shape ----------' , self.X_valid.shape )

    def set_historical_data(self, historical_data):
        self.prepare_data(historical_data)

    def build_model(self, show_loss_results=True, verbose=0, get_architecture=False, trails = None, fast = False):# setup_dict, historical_data_lst):
        # best_model = get_best_model(setup_dict, historical_data_lst)
        if fast:
            model_architecture_str = self.build_fast_model()
        else:
            model_architecture_str = self.get_best_architecture()

        if get_architecture:print(self.get_model_architecture_str())
        self.model_h5_str, train_loss, val_loss, ep_train_valid_str = self.get_best_random_model(show_loss_results=show_loss_results, verbose = verbose, trails = trails)
        self.model = load_model(self.model_h5_str)

        fit_str = str(train_loss)[:6] + '_' + str(val_loss)[:6]
        return model_architecture_str, self.model_h5_str, fit_str, ep_train_valid_str

    def load_model(self, model_str):
        self.state = 'loaded'
        self.set_model_str( model_str )
        self.model = load_model( model_str )
        self.get_model_architecture_from_external_model()
        print('Loading model: %s ***********'%model_str)
        # print('Loaded model Architecture:\t', ' '*25+self.get_model_architecture_str().replace('\n','\n'+' '*25))
        return self.get_model_architecture_str(), self.model_h5_str, 'Loaded_model'

    def fit(self, show_loss_results = False, verbose = 0 ): # original
        '''
            fits the model once and return model results and model_h5_str
        '''
        if len(self.X_train)< 1:
            assert False,'Err:004A: Data is not enough to train/fit the model'
        elif len(self.X_valid) < 1:
            assert False,'Err:004B: Data is not enough to validate the model'

        model = self.get_initinal_model()
        call_back_lst, model_h5_str = self.get_call_back_lst()
        history = model.fit(
                                    x = self.X_train,
                                    y = self.y_train ,
                                    batch_size = self.best_architecture_dict['batch_size'],
                                    epochs = self.best_architecture_dict['nb_epochs'],
                                    verbose = verbose,
                                    callbacks = call_back_lst,
                                    validation_data=(self.X_valid, self.y_valid),
                                    shuffle=False
                            )
        loss_history_str = 'train: '+ get_lst_str(history.history['loss'])+'\n'+ ' '*35 + 'valid: '+ get_lst_str(history.history['val_loss'])

        ind = history.history['val_loss'].index(np.min(history.history['val_loss']))
        # ind = len(history.history['val_loss'])-1
        val_loss = str(history.history['val_loss'][ind])[:7]
        train_loss = str(history.history['loss'][ind])[:7]
        if show_loss_results:print( 'train_loss:', train_loss, '  val_loss:', val_loss, '=>', '%-7s'%(str(float(val_loss)-float(train_loss))[:7]) , '  ',model_h5_str)

        K.clear_session()
        del model
        gc.collect()
        import keras.backend.tensorflow_backend
        if keras.backend.tensorflow_backend._SESSION:
            tf.reset_default_graph()
            keras.backend.tensorflow_backend._SESSION.close()
            keras.backend.tensorflow_backend._SESSION = None

        return [model_h5_str, train_loss, val_loss, loss_history_str]

    def update(self, data):
        # model = load_model(self.get_model_str())
        call_back_lst, _ = self.get_call_back_lst()
        model_h5_str = self.get_model_str()[:-3] + '_refit.h5'
        if len(data)<= self.setup_['LSTM_w']:
            assert False,'Err:007A:Data is not enough to update the model'
        elif len(data)*self.setup_['train_validate_ratio'] < 1:
            assert False,'Err:007B:Data is not enough to train-validation split the data to update the model'

        x_data, y_data = get_data_look_back(dataseries = data,
                                            Apply_diff = False,
                                            window_size = self.setup_['LSTM_w'])
        self.X_train, self.y_train, self.X_valid, self.y_valid = train_test_split(x_data, y_data, self.setup_['train_validate_ratio'])
        history = self.model.fit(
                                        x = self.X_train,
                                        y = self.y_train ,
                                        batch_size = self.best_architecture_dict['batch_size'],
                                        epochs = 2,
                                        verbose = 0,
                                        shuffle=False,
                                        callbacks= call_back_lst,
                                        validation_data=(self.X_valid, self.y_valid),
                                )
        # print('Upd train: ', get_lst_str(history.history['loss']),'\nUpd valid: ', get_lst_str(history.history['val_loss']),sep='')
        ind = history.history['val_loss'].index(np.min(history.history['val_loss']))
        val_loss = history.history['val_loss'][ind]
        train_loss = history.history['loss'][ind]
        fit_str = str(train_loss)[:6] + '_' + str(val_loss)[:6]
        self.set_model_str(model_h5_str)
        self.model.save(self.get_model_str())
        return fit_str
		
    def upgrade_model(self, new_historical_data, show_loss_results=True):
        self.prepare_data(new_historical_data, Apply_diff=False)
        return self.build_model(show_loss_results=show_loss_results)

    def get_model_architecture_from_external_model(self):
        self.best_architecture_dict = {}
        self.best_architecture_dict['lstm_cells'] = self.model.layers[1].get_config()['units']
        self.best_architecture_dict['d1_nodes'] = self.model.layers[2].get_config()['units']
        self.best_architecture_dict['activation_1'] = self.model.layers[2].get_config()['activation']

        if self.model.layers[3].name == 'denes_2' :
            self.best_architecture_dict['d2_nodes'] = self.model.layers[3].get_config()['units']
            self.best_architecture_dict['activation_2'] = self.model.layers[3].get_config()['activation']
        else:
            self.best_architecture_dict['d2_nodes'] = 0
            self.best_architecture_dict['activation_2'] = ''

        if self.model.layers[3].name == 'dropout_1':
            self.best_architecture_dict['dropout1'] = self.model.layers[3].get_config()['rate']
        elif self.model.layers[4].name == 'dropout_1':
            self.best_architecture_dict['dropout1'] = self.model.layers[4].get_config()['rate']

        self.best_architecture_dict['batch_size'] = 0

    def get_model_str(self):
        return self.model_h5_str

    def set_model_str(self, model_str):
        self.model_h5_str = model_str

    def get_call_back_lst(self):
        model_h5_str = 'models/%s.h5'%(get_TimeStamp_str() + str(time.time()))
        checkpoint = ModelCheckpoint(filepath=model_h5_str,monitor='val_mean_absolute_error',verbose=0,save_best_only=True,mode='min')
        # early_stopping = EarlyStopping(monitor='val_mean_absolute_error',patience=4,verbose=0,mode='min')
        # reduceLR_On_Plateau = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor = 0.15, patience = 2,
        #     mode = "auto", epsilon = 1e-04, cooldown = 0,  min_lr = 0)
        # , reduceLR_On_Plateau
        return [checkpoint], model_h5_str # , early_stopping

    def get_best_random_model(self, trails = None, show_loss_results = False, verbose = 0 ):
        res = []
        up_to = trails if trails is not None else self.setup_['number_random_fits']
        for _ in range(up_to):
            res.append(self.fit(show_loss_results = show_loss_results, verbose = verbose))

        self.set_model_str(sort_tuble(res)[0])
        # print( 'best validation model: train_loss:', sort_tuble(res)[1], '  val_loss:', sort_tuble(res)[2])
        return sort_tuble(res)

    def get_model_architecture_str(self):
        s = '\n--------------------------\nLSTM:%d\ndenes_1:%d Activation:%s\ndenes_2:%d Activation:%s\nDropout:%-.1f  batch_size:%d'%(
                self.best_architecture_dict['lstm_cells'],
                self.best_architecture_dict['d1_nodes'],
                self.best_architecture_dict['activation_1'],
                self.best_architecture_dict['d2_nodes'],
                self.best_architecture_dict['activation_2'],
                self.best_architecture_dict['dropout1'],
                self.best_architecture_dict['batch_size']
                )
        try:s+='  epoches:%d'%self.best_architecture_dict['nb_epochs']
        except:pass
        s += '\n--------------------------'
        return s

    def get_model_summary(self):
        print(self.model.summary())

    def get_initinal_model(self):
        assert self.best_architecture_dict is not None, '\nmodel architecture is not initialized yet'
        p_epochs=int(self.best_architecture_dict['nb_epochs'])
        lstm_cells =int(self.best_architecture_dict['lstm_cells'])
        d1_nodes = int(self.best_architecture_dict['d1_nodes'])
        d2_nodes = int(self.best_architecture_dict['d2_nodes'])
        p_dropout = self.best_architecture_dict['dropout1']
        act_1 = self.best_architecture_dict['activation_1']
        act_2 = self.best_architecture_dict['activation_2']
        batch_size = int(self.best_architecture_dict['batch_size'])
        input_layer = Input(shape=( self.X_train.shape[1], self.X_train.shape[2]))
        out_model = LSTM(lstm_cells, return_sequences=False)(input_layer)
        out_model = Dense(d1_nodes, activation=act_1, name='denes_1')(out_model)
        if d2_nodes:
            out_model = Dense(d2_nodes, activation=act_2, name='denes_2')(out_model)
        out_model = Dropout(p_dropout) (out_model)
        out_model = Dense(1, activation='linear')(out_model)
        model = Model( inputs=[input_layer], outputs = [out_model] )
        model.compile( loss = 'mae', optimizer = 'adam', metrics=['mae'] )
        return model

    def get_best_architecture(self, trails = None):
        def create_model_hyperopt(params):
            K.clear_session()
            p_epochs=int(params['nb_epochs'])
            lstm_cells =int(params['lstm_cells'])
            d1_nodes = int(params['d1_nodes'])
            d2_nodes = int(params['d2_nodes'])
            p_dropout = params['dropout1']
            act_1 = params['activation_1']
            act_2 = params['activation_2']
            batch_size = int(params['batch_size'])
            start_time = time.time()

            # use 'try except', in case some parameters are wrong, the script will fail and stop
            try:
                input_layer = Input(shape=( self.X_train.shape[1], self.X_train.shape[2]))
                out_model = LSTM(lstm_cells, return_sequences=False)(input_layer)
                out_model = Dense(d1_nodes, activation=act_1, name='denes_1')(out_model)
                if d2_nodes:
                    out_model = Dense(d2_nodes, activation=act_2, name='denes_2')(out_model)
                if p_dropout:
                    out_model = Dropout(p_dropout) (out_model)
                out_model = Dense(1, activation='linear')(out_model)

                model = Model( inputs=[input_layer], outputs = [out_model] )
                model.compile( loss = 'mse', optimizer = 'adam', metrics=['mae'] )
                history = model.fit(
                                                x = self.X_train,
                                                y = self.y_train ,
                                                batch_size = batch_size,
                                                epochs = p_epochs,
                                                verbose = 0,
                                                # callbacks = [self.get_call_back_lst()[0][1]],
                                                validation_data=(self.X_valid, self.y_valid),
                                                shuffle=False
                                        )

                ind = history.history['val_loss'].index(np.min(history.history['val_loss']))
                val_loss = history.history['val_loss'][ind]

                end_time = time.time()
                execution_time = end_time - start_time

                K.clear_session()
                del model
                gc.collect()
                import keras.backend.tensorflow_backend
                if keras.backend.tensorflow_backend._SESSION:
                    tf.reset_default_graph()
                    keras.backend.tensorflow_backend._SESSION.close()
                    keras.backend.tensorflow_backend._SESSION = None

                return {'loss': val_loss,
                        'status': STATUS_OK,
                        'history': history.history,
                        'params': params,
                        'execution_time': execution_time}


            except:
                print('\n**** Err A004: Hyperopt:\n',traceback.format_exc())
                import sys;sys.exit()
            return {'loss': 0, 'status': STATUS_FAIL}

        search_space = self.setup_['hyperopt_space']
        max_trials = trails if trails is not None else self.setup_['hyperopt_max_trials']
        best = fmin( fn = create_model_hyperopt,
                     space = search_space,
                     algo = tpe.suggest,
                     max_evals = max_trials,
                     trials = Trials())
        self.best_architecture_dict = hyperopt.space_eval(space = search_space,
                                                     hp_assignment = best)
        return self.get_model_architecture_str()