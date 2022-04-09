# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg>
@brief: config file
"""

import os
import numpy as np
import matplotlib.pylab as plt
import traceback
import pickle
from ._utils import get_setup_data, load_mote_data, read_Intel_data, read_Benzene_data
from ._time_utils import get_TimeStamp_str


def load_experiment(exp_id, summary=True): # i.e., load_experiment("exp/20191202_10_0050AM.pkl")
    try:
        # file_name = exp_id if exp_id[-4:]=='.pkl' else exp_id+'.pkl'
        exp_id = 'Experiments/' + exp_id if 'Experiments/' not in exp_id else exp_id
        # print( 'exp_id 222 = ' , exp_id )
                
        with open( exp_id, "rb") as f:
            ex = pickle.load(f)
        if summary:print(ex.summary())
        return ex
    except Exception as exc:
        print('\n**** Err:005G Loading Experiment Err:\n',traceback.format_exc())
        return False

def re_run_experiment(exp_id, setup_=None, historical_data=None, test_data=None, dump_exp=False):

    exp = load_experiment(exp_id, summary=False)
    if setup_ == None:
        setup_ = exp.setup_
        
    if 'Intel-DS' in exp_id:
        df = read_Intel_data()
        
        # Intel Experiments were run using historical data, therefore, the model is trained and refitted using the gathering period data (initial phase + LMS phase). and, hence, refitted ('_refit.h5') version of the model is used.
        model = exp.phase['phase_2']['model_h5_str']
        if os.path.isfile(model[:-3]+'_refit.h5'):
            model=model[:-3]+'_refit'+'.h5'
        else:
            assert False,'Err:001G:model "%s" is not found '%model            
        
        exp_str = int(exp.exp_str)

        if historical_data == None:
            historical_data =  load_mote_data( df, moteID = exp_str, date='2004-03-05' )
        if test_data == None:
            test_data = load_mote_data( df, moteID = exp_str, date= ('2004-03-06', '2004-03-09') ).values
            
    elif 'Benzene-DS' in exp_id:
        test_data = read_Benzene_data()
        model = []
        exp_str = 'Benzene'
        # get phases' models
        for j in range(2, len(exp.phase)):
            model.append(exp.phase['phase_%d'%j]['model_h5_str'])
        
        
    
    # for k,v in setup_.items():
        # print(k,v)
    from ._run import run_exp
    a = run_exp(setup_, exp_variable='e_max_threshold', test_data=test_data ,historical_data=historical_data, exp_str=exp_str,
                    model=model, dump_exp=dump_exp)
    # experiment = load_experiment(a[0],summary=True)
    return a


class Experiment:
    def __init__(self, setup_dict, exp_str=''):
        self.phase = {} # a dictionary of phase_ID and its corresponding tuple data (obs_count,)
        self.total_miss_pred_count = 0
        self.temp = None  # the sensed_data being observed by the sensor i.e., temperature, humidity, ..., etc
        self.agg_lst = []
        self.agg_lst_true = []
        self.key = None # most top phase's key
        self.setup_ = setup_dict
        self.block_err_count = 0
        self.exp_id = '%-19s'%get_TimeStamp_str()
        self.exp_str = exp_str  # experiment's details/info, i.e., moteID, datasetname, ..., etc.
        self.experimental_results = ''
        self.experiment_setup = get_setup_data(setup_dict)
        self.mae = 0.0
        self.mse = 0.0

    def set_temp(self, x):
        self.temp = x

    def get_temp(self):
        return self.temp
    def increment_temp(self, value):
        self.temp += value
    def get_agg_lst(self, last=None):
        if last:
            return self.agg_lst[-last:]
        else:
            return self.agg_lst
    def set_miss_pred_count(self, count):
        self.total_miss_pred_count += count
        self.block_err_count += count
        self.phase[self.key]['No_of_observations'] += count
        self.phase[self.key]['miss_pred_count'] += count
    def get_setup_dict_str(self):
        print(  'Apply_diff:'+str(self.setup_['Apply_diff'])+'\n'+ \
                'train_on_historical_data:'+str(self.setup_['train_on_historical_data'])+'\n'+ \
                'gathering_period:'+str(self.setup_['gathering_period'])+'\n'+ \
                'Update_method:'+str(self.setup_['block_size'])+'_'+str(self.setup_['block_err_threshold'])+'_'+str(self.setup_['phase_accumulated_block_err'])+'\n'+ \
                'data_size:'+str(self.setup_['data_size'])+'\n'+ \
                'e_max_threshold:'+str(self.setup_['e_max_threshold'])+'\n'+ \
                'window_size:'+str(self.setup_['LSTM_w'])+'\n'+ \
                'train_validate_ratio:'+str(self.setup_['train_validate_ratio'])+'\n'+ \
                'number_random_fits:'+str(self.setup_['number_random_fits'])+'\n'+ \
                'hyperopt_max_trials:'+str(self.setup_['hyperopt_max_trials']))
    def new_phase(self, starting_index, phase_title='', phase_model_architecture='', model_h5_str='', model_fit_str = '', loss_history_str = '', build_time = 0):
        if self.key:self.update_phase_mse()
        self.key = 'phase_'+str(len(self.phase))
        self.phase[self.key] = {

                                # phase info:
                                    'starting_index':starting_index,
                                    'phase_title':phase_title,

                                # model parameters:
                                    'phase_model_architecture':' '*25+phase_model_architecture.replace('\n','\n'+' '*25),
                                    'model_h5_str':model_h5_str,
                                    'model_fit_str':model_fit_str,
                                    'loss_history_str':loss_history_str,
                                    'build_time':build_time,

                                # phase experiment's summary parameters
                                    'No_of_observations':0,
                                    'miss_pred_count':0,
                                    'squared_error':0,
                                    'absolute_error':0,
                                    'mse':0,
                                    'mae':0,
                                    'accumulated_err_ratio':[],
                                    'miss_pred_index':[],
                                    'block_err':[],
                                }
        self.block_err_count = 0

    def get_miss_pred_indices(self, phase_id):
        if phase_id in self.phase.keys():
            return self.phase[phase_id]['miss_pred_index']
        else:
            print(phase_id, 'is not found ...')

    # set the aggregate list
    def set_agg_lst(self, lst):
        if len(self.phase)==1:
           self.phase[self.key]['accumulated_err_ratio'] = np.ones(len(lst))
        self.agg_lst = lst[1:]

        if self.setup_['Apply_diff']:
            for i in range( 1, len(lst)+1 ):
                self.agg_lst_true.append( np.sum(lst[:i]) )

        self.set_miss_pred_count(len(lst))

    # add a record/observation to the aggregate list
    def push_observation(self, item):
        self.agg_lst.append( item )
        self.agg_lst_true.append( self.temp )


    def set_phase_model_architecture(self,phase_model_architecture):
        self.phase[self.key]['phase_model_architecture'] = phase_model_architecture

    def add_correct_pred_obs(self, y_pred, mae_value):
        # y_pred: the predicted difference, that is accepted
        # mae_value: is the mean absolute error between the actual reading and the predicted one

        mae_value = np.abs(mae_value)
        if mae_value<0:
            print('err 00H7', mae_value)#qqqq

        # increase "No_of_observations" for that phase
        self.increment_phase_obs_counts()
        
        self.push_observation(y_pred)
        self.increment_temp(y_pred) # handling temp value

        self.phase[self.key]['squared_error'] += mae_value * mae_value



        self.phase[self.key]['absolute_error'] += mae_value
        self.phase[self.key]['accumulated_err_ratio'].append( self.phase[self.key]['miss_pred_count']/self.phase[self.key]['No_of_observations']   )
        if self.check_for_block_end():
            return self.check_for_phase_update()
        else:
            return False

    def add_miss_pred_obs(self, true_difference, true_temp, err_value):
        # true_difference: is the correct difference
        # true_temp: is the actual sensed temp

        # increase "No_of_observations" for that phase
        self.increment_phase_obs_counts()

        if self.setup_['Apply_diff']:self.push_observation(true_difference)
        else:self.push_observation(true_temp)

        self.set_temp(true_temp) # handling temp value

        self.phase[self.key]['accumulated_err_ratio'].append( self.phase[self.key]['miss_pred_count']/self.phase[self.key]['No_of_observations']   )

        self.block_err_count += 1
        self.total_miss_pred_count += 1
        self.phase[self.key]['miss_pred_count'] +=1
        self.phase[self.key]['miss_pred_index'].append( (self.phase[self.key]['No_of_observations'], err_value) )
        if self.check_for_block_end():
            return self.check_for_phase_update()
        else:
            return False

    def increment_phase_obs_counts(self):
        self.phase[self.key]['No_of_observations'] +=1

    def check_for_phase_update(self):
        tmp = np.array(self.phase[self.key]['block_err'][-self.setup_['phase_accumulated_block_err']:])[:,1]
        if len(tmp) < self.setup_['phase_accumulated_block_err']:
            pass
        elif np.sum(tmp) == len(tmp):
            return True
        return False

    def check_for_block_end(self):
        if self.phase[self.key]['No_of_observations'] % self.setup_['block_size'] == 0:

            Block_not_valid = 1 if self.block_err_count >= self.setup_['block_err_threshold'] else 0

            #                                           Block_error_count,  Block_not_valid (Y/N)
            self.phase[self.key]['block_err'].append( [self.block_err_count,  Block_not_valid] )

            self.block_err_count = 0
            return True
        return False

            # check for update: it maybe a add_correct_pred_obs but we need to make an update
            # block_size, block_error_threshold

    def get_last_phase_obs_count(self):
        return self.phase[self.key]['No_of_observations']

    def update_all_mse(self):
        try:
            for i in self.phase.keys():
                self.phase[i]['mse'] = self.phase[i]['squared_error']/self.phase[i]['No_of_observations']
                self.phase[i]['mae'] = self.phase[i]['absolute_error']/self.phase[i]['No_of_observations']
        except Exception as exc:pass

    def update_phase_mse(self):
        try:
            self.phase[self.key]['mse'] = self.phase[self.key]['squared_error']/self.phase[self.key]['No_of_observations']
            self.phase[self.key]['mae'] = self.phase[self.key]['absolute_error']/self.phase[self.key]['No_of_observations']
        except Exception as exc:pass

    # get experiment summary
    def summary(self, get_str = False):
        try:
            if self.phase[self.key]['mse']==0.0:self.update_phase_mse()  # update 'mse' of the last phase
        except Exception as exc:pass

        try:
            if self.mse == 0.0:self.get_final_mse()
        except Exception as exc:pass

        s = ''
        s += '\n=====================================' + '\n'
        s += 'No_of_phases: ' + str(len(self.phase)) + '\n'
        for i in range(len(self.phase)):
            s += '\tPhase_%-2d *** title: "%s"'%(i, self.phase['phase_'+str(i)]['phase_title']) + '\n'
            for k,v in self.phase['phase_'+str(i)].items():
                if k in['phase_title','loss_history_str','miss_pred_index']:
                    continue
                elif k=='accumulated_err_ratio':
                    s += '\t\t '+ k + ' : [' + ', '.join(['%-.5f'%i for i in v[-7:]]) + ']\n'
                else:
                    if str(v).strip() == '':v = ' ---' 
                    s += '\t\t '+ k + ' :'+ str(v) + '\n'
            s += '\n'

        try:s += 'total_miss_pred_count: '+ str(self.total_miss_pred_count) + '\n'
        except Exception as exc:pass

        try:s += 'MAE: %-.5f'%self.mae + '\n'
        except Exception as exc:pass

        try:s += 'MSE: %-.5f'%self.mse + '\n'
        except Exception as exc:pass

        try:s += 'Experimental results : '+ str(self.experimental_results) + '\n'
        except Exception as exc:s += 'Experimental results : Not found' + '\n'

        try:s += 'Experimental setup   : '+ str(self.experiment_setup) + '\n'
        except Exception as exc:s += 'Experimental setup     : Not found' + '\n'

        s += '=====================================\n' + '\n'
        if get_str: return s
        else: print(s)

    def get_mean_error(self):
        if self.mse == 0.0:self.get_final_mse()
        return 'MAE: %-.5f  '%self.mae + 'MSE: %-.5f'%self.mse

    def get_final_mse(self):
        sum_mse, sum_mae = 0,0
        for i in self.phase.keys():
            # print( i , self.phase[i]["squared_error"], self.phase[i]['absolute_error'],  self.phase[i]['No_of_observations'])
            sum_mse += self.phase[i]['squared_error']
            sum_mae += self.phase[i]['absolute_error']

        # print( 'final---', sum_mse, len(self.agg_lst),' END: ', float(sum_mse / len(self.agg_lst) ) )
        # print( 'final---', sum_mae, len(self.agg_lst),' END: ', float(sum_mae / len(self.agg_lst) ) )
        self.mae = float(sum_mae / len(self.agg_lst) )
        self.mse = float(sum_mse / len(self.agg_lst) )

        return self.mae, self.mse

    def set_exp_results(self,str_results):
        self.experimental_results = str_results

    def plot_accumulated_err_ratio(self, v_space = 0.2, fig_size = None, exclude_starting = 2):
        if fig_size==None:fig_size=(12, len(self.phase)*5)
        plt.rcParams.update({'figure.figsize':fig_size, 'figure.dpi':120})

        if len(self.phase) - exclude_starting ==1:
            fig, axes = plt.subplots(1)
            i = len(self.phase)-1
            axes.plot( list(range(self.phase['phase_%d'%i]['No_of_observations']))[0:],
                              self.phase['phase_%d'%i]['accumulated_err_ratio'][0:])
            axes.tick_params(labelsize=10)
            axes.set_title('phase_%d: '%i + self.phase['phase_'+str(i)]['phase_title'] + \
                              '     No_of_Obs: %d'%self.phase['phase_'+str(i)]['No_of_observations'] + \
                              '     Miss_pred: %d'%self.phase['phase_'+str(i)]['miss_pred_count']
                              )
            plt.show()
        else:
            fig, axes = plt.subplots(len(self.phase) - exclude_starting)
            fig.subplots_adjust(hspace=v_space)
            # , plt_all = False
            # start_inx = 0 #if plt_all else self.setup_['min_to_update']
            for i in range(exclude_starting, len(self.phase)):
                axes[i - exclude_starting].plot( list(range(self.phase['phase_'+str(i)]['No_of_observations']))[0:],
                              self.phase['phase_'+str(i)]['accumulated_err_ratio'][0:]
                             )
                axes[i - exclude_starting].tick_params(labelsize=10)
                axes[i - exclude_starting].set_title('phase_%d: '%i + self.phase['phase_'+str(i)]['phase_title'] + \
                                  '     No_of_Obs: %d'%self.phase['phase_'+str(i)]['No_of_observations'] + \
                                  '     Miss_pred: %d'%self.phase['phase_'+str(i)]['miss_pred_count']
                                  )
            plt.show()

    def plot_phase_bins(self, v_space = 0.2, fig_size = None, exclude_starting = 2):
        if fig_size==None:fig_size=(12, len(self.phase)*5)
        plt.rcParams.update({'figure.figsize':fig_size, 'figure.dpi':120})

        fig, axes = plt.subplots(len(self.phase)-exclude_starting) # [:1])
        fig.subplots_adjust(hspace=v_space)
        start_inx = 0 #if plt_all else self.setup_['min_to_update']
        for i in range(exclude_starting, len(self.phase)):
            axes[i - exclude_starting].plot( list(range(self.phase['phase_'+str(i)]['No_of_observations']))[start_inx:],
                          self.phase['phase_'+str(i)]['accumulated_err_ratio'][start_inx:]
                         )
            axes[i - exclude_starting].tick_params(labelsize=10)
            axes[i - exclude_starting].set_title('phase_%d: '%i + self.phase['phase_'+str(i)]['phase_title'] + \
                              '     No_of_Obs: %d'%self.phase['phase_'+str(i)]['No_of_observations'] + \
                              '     Miss_pred: %d'%self.phase['phase_'+str(i)]['miss_pred_count']
                              )
        plt.show()

    def plot_loss(self,phase_id = 'phase_2'):
        if self.phase[phase_id]['loss_history_str'] == '':
            print('No loss data')
        else:
            loss = self.phase[phase_id]['loss_history_str'].split('\n')
            # print( 'loss = ' , self.phase[phase_id]['loss_history_str'] )
            train = loss[0][7:].replace(' ','').split(',')
            valid = loss[1].strip()[7:].replace(' ','').split(',')
            # print( 'train = ' , train )
            # print( 'valid = ' , valid )
            train = [float(i) for i in train]
            valid = [float(i) for i in valid]

            x = list(range(1, len(train)+1))
            plt.plot(x, train, 'r.-', label='Train Loss')
            plt.plot(x, valid, 'b.-', label='Validation Loss')
            plt.legend(loc="upper right")#, markerscale=2., scatterpoints=180, fontsize=40)
            plt.show()

    def dump(self, dir_name = 'Experiments'):
        import copy
        try:
            ob = copy.deepcopy(self)
            with open(dir_name+'/'+str(self.exp_id)+'.pkl', "wb") as f:
                pickle.dump(ob, f, pickle.HIGHEST_PROTOCOL )
            print('successfully dump to "'+dir_name+'/%s.pkl"'% self.exp_id)
        except FileNotFoundError:
            import os;os.mkdir(dir_name)
            try:
                with open(dir_name+'/'+str(self.exp_id)+'.pkl', "wb") as f:
                    pickle.dump(ob, f, pickle.HIGHEST_PROTOCOL)
                print('successfully dump to "'+dir_name+'/%s.pkl"'% self.exp_id)
            except Exception as exc:
                print('\n**** Err:005A Dumping Experiment Err:\n',traceback.format_exc())

    #def get_accumulated_err_ratio(self):
        #    return self.phase[self.key]['accumulated_err_ratio'][-1]

# plot_phase_bins
