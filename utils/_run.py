# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg>
@brief: config file
"""

from keras import backend as K
import numpy as np
import traceback
try:
    import padasip as pa
except Exception as exc:
    print('\n**** Err A001 :\n', traceback.format_exc())
import time
import gc
import operator

from ._model import Custom_Model
from ._experiment import Experiment
from ._write_to_file import write_to_file, dump

from ._utils import get_setup_data
from utils._time_utils import get_timestamp_m

def run_experiment(setup_, exp_variable, test_data, historical_data = None, exp_str='', load_model=None,
                   show_loss_results=False,get_summary=False, plt=False, upgrade_verbose=False,
                   runs=5, file='exp.txt', print_res=False, progress='', write_all = False, dump_exp=True ):
    ls = []
    a = time.time()
    if load_model != None:runs=1
    print("\r", '%d/%d   '%(0, runs),get_timestamp_m(), '  ', progress,end="")
    for i in range(runs):
        tmp = run_exp(setup_, exp_variable, test_data=test_data ,historical_data=historical_data, exp_str=exp_str,
                          show_loss_results=show_loss_results,get_summary=get_summary, plt=plt, upgrade_verbose=upgrade_verbose, dump_exp=dump_exp)
        ls.append(tmp)
        if write_all:dump(' '.join(tmp)+'\n',file )
        print("\r", '%d/%d   '%(i+1, runs),get_timestamp_m(), '  ', progress,end="")

    for i in sorted(ls, key=operator.itemgetter(1), reverse=True):
        dump('\n'+' '.join(i),file)
        if print_res:print(i)
    dump('      '+ str((time.time()-a)/60)[:4]+'\n\n',file )

def run_exp(setup_,exp_variable, test_data, historical_data = None, exp_str='', model=None,
            show_loss_results=False,get_summary=False, plt=False, upgrade_verbose=False, dump_exp=True):
    exp_start_time = time.time()
    object_ = Custom_Model(setup_)
    
    if model != None:
        if isinstance(model, list): # multi phases of LSTM 
            model_architecture, model_h5_str, fit_str = object_.load_model(model[0])
            model = model[1:] # remove first index model from the models list
        elif isinstance(model, str): # single phase of LSTM 
            model_architecture, model_h5_str, fit_str = object_.load_model(model)
        loss_history_str = ''
        build_time = 0
        

    # if historical_data == None:setup_['train_on_historical_data']=False
    total_time = 0.0
    ex = Experiment(setup_, exp_str=exp_str)
    ex.new_phase(starting_index=0, phase_title='initializing')
    
    # 1. train_on_historical_data
    if setup_['train_on_historical_data'] and object_.state != 'loaded':
        # get one-day-before historical data
        object_.set_historical_data(historical_data)
        start_time = time.time()
        model_architecture, model_h5_str, fit_str, loss_history_str = object_.build_model(show_loss_results=show_loss_results)
        build_time = time.time() - start_time
        
    if setup_['Apply_diff']:
        lst = [test_data[0]]
        if setup_['initialization_period'] == None:
            for i in range(setup_['LMS_w']):lst.append( test_data[i+1]-test_data[i] )
        else:
            for i in range(setup_['initialization_period']):lst.append( test_data[i+1]-test_data[i] )
        ex.set_temp( np.sum(lst) )
    else:
        lst =  test_data[: setup_['LMS_w']+1 ]
        ex.set_temp( test_data[setup_['LMS_w']] )

    ex.set_agg_lst( lst )

    # 2. LMS_period
    if setup_['LMS_period']>0:
        ex.new_phase(starting_index=setup_['LMS_w']+1, phase_title='LMS')
        try:
            level_1_st  = setup_['LMS_w']+1
            level_1_end = level_1_st + setup_['LMS_period']
            filt = pa.filters.FilterLMS(3, mu=.01,w=[ 0.88744788, -0.28791624,  0.52442267])

            # run the LMS on the data in the initialization phase, to learn the data pattern
            if len(ex.agg_lst) > setup_['LMS_w']:
                for i in range(len(ex.agg_lst)-setup_['LMS_w'] + 1):
                    _ = filt.predict(ex.agg_lst[i:i+setup_['LMS_w']])

            # start employing the LMS for prediction
            for i in range( level_1_st, level_1_end ):
                y_pred = filt.predict(ex.get_agg_lst(last=3))
                if setup_['Apply_diff']==True:new_temp = (ex.get_temp() + y_pred)
                else: new_temp = y_pred
                if np.abs( test_data[i] - new_temp ) < setup_['e_max_threshold']:
                    ex.add_correct_pred_obs( y_pred = y_pred,
                                             mae_value = (test_data[i] - new_temp) )
                else:
                    ex.add_miss_pred_obs( true_difference = test_data[i] - ex.get_temp(),
                                          true_temp = test_data[i],
                                          err_value = test_data[i] - new_temp )
        except Exception as exc:
            print('**** Err Z002: LMS \n',traceback.format_exc())
            
    # 3. train/update the model
    if object_.state != 'loaded':
        try:
            if setup_['train_on_historical_data']: # update the model
                update_str_1 = object_.update( ex.get_agg_lst() )
                fit_str = fit_str + '_refit_' + update_str_1
            else: # build the model from scratch
                object_.set_historical_data( ex.get_agg_lst() )
                start_time = time.time()
                model_architecture, model_h5_str, fit_str, loss_history_str = object_.build_model(show_loss_results=show_loss_results,
                                                                                                  fast = True,
                                                                                                  get_architecture = False)
                build_time = time.time() - start_time
        except Exception as exc:
            print('**** Err Z003: updating \n',traceback.format_exc())

    # 4. deploying of the dl model
    try:
        ex.new_phase(starting_index=level_1_end+1,
                     phase_title='LSTM_starting',
                     phase_model_architecture = model_architecture,
                     model_h5_str = model_h5_str,
                     model_fit_str = fit_str,
                     loss_history_str = loss_history_str,
                     build_time = build_time)

        level_2_end = len(test_data)
        for i in range( level_1_end, level_2_end ):
            x_test_instance = np.reshape(ex.get_agg_lst(last=setup_['LSTM_w']), (1, setup_['LSTM_w'], 1))

            a = time.time()
            y_pred = object_.model.predict(x_test_instance)[0][0]
            total_time = time.time() - a

            if setup_['Apply_diff']==True:new_temp = (ex.get_temp() + y_pred)
            else: new_temp = y_pred

            if np.abs( test_data[i] - new_temp ) < setup_['e_max_threshold']:
                need_update = ex.add_correct_pred_obs( y_pred = y_pred,
                                                       mae_value = (test_data[i] - new_temp) )
            else:
                need_update =  ex.add_miss_pred_obs( true_difference = test_data[i] - ex.get_temp(),
                                                     true_temp = test_data[i],
                                                     err_value = test_data[i] - new_temp )
            if setup_['employ_update_policy']==True:
                if need_update:

                    K.clear_session()
                    del object_.model
                    gc.collect()
                    import keras.backend.tensorflow_backend
                    if keras.backend.tensorflow_backend._SESSION:
                        import tensorflow as tf
                        tf.reset_default_graph()
                        keras.backend.tensorflow_backend._SESSION.close()
                        keras.backend.tensorflow_backend._SESSION = None
                    
                    ind_ = setup_['LSTM_w']+level_1_end+i+1
                    
                    
                    if object_.state == 'loaded':
                        # load the next model in the model parameter list which has index of 0, as we remove the first index each time we load a model from the model parameter
                        object_.load_model( model[0] )
                        
                        model_h5_str = model[0]
                        model_architecture = object_.get_model_architecture_str()
                        upgrade_str, loss_history_str = '', None
                        model = model[1:] # remove first index of model from the models list
                    else:
                        # upgrade the model
                        start_time = time.time()
                        model_architecture, model_h5_str, upgrade_str, loss_history_str = object_.upgrade_model( ex.get_agg_lst(),
                                                                                                                show_loss_results=show_loss_results )
                        build_time = time.time() - start_time
                        if upgrade_verbose:print('-------- Upgrading ------ iteration:%d  progress:%-.2f%%'% (i, i*100/(level_2_end+level_1_end)),
                                                 str(build_time)[:5],' ', ex.phase[ex.key]['block_err'][-3:],' ',get_timestamp_m())
                    
                    ex.new_phase(starting_index=ind_,
                                 phase_title='Update_'+ str(ind_),
                                 phase_model_architecture = model_architecture,
                                 model_h5_str = model_h5_str,
                                 model_fit_str = upgrade_str,
                                 loss_history_str=loss_history_str,
                                 build_time = build_time)


        K.clear_session()
        del object_
        gc.collect()
        import keras.backend.tensorflow_backend
        if keras.backend.tensorflow_backend._SESSION:
            import tensorflow as tf
            tf.reset_default_graph()
            keras.backend.tensorflow_backend._SESSION.close()
            keras.backend.tensorflow_backend._SESSION = None
    except Exception as exc:
        print('**** Err Z004: LSTM Phase encounters an error\n',traceback.format_exc())
        
    reduction = (1-ex.total_miss_pred_count/len(test_data))*100
    exp_time = str( (time.time() - exp_start_time)/60)[:5]

    res = 'W_size:%-2d/%-2d mote_ID:%-2s '%(setup_['LMS_w'],setup_['LSTM_w'], exp_str)+ \
                      ' Reduc:%s%% '%(str(reduction)[:5])+ \
                      ' fit_%s '%fit_str +\
                      ' time:%s'%exp_time +\
                      ' phases:%s'%str(len(ex.phase))
                      # ' MSE_%s '%str( ex.get_mse() )[:6]+\
    ex.set_exp_results(res)
    
    # write_to_file(ex.exp_id +'\t'+get_setup_data(setup_), res, ex.summary(get_str=True)[1:])
    if get_summary:ex.summary()
    if plt:ex.plot_accumulated_err_ratio()
    
    print(ex.summary())
    if dump_exp:ex.dump()
    
    return (ex.exp_id, '%-6s%%'%str(reduction)[:6], '%-4d'%ex.total_miss_pred_count, '%2s'%str(exp_str)+'_'+ str(ex.setup_[exp_variable]), '#Phases:'+str(len(ex.phase)), fit_str,'total_time', str(total_time), 'avg_%-.3f'%(total_time / (level_2_end - level_1_end)))
