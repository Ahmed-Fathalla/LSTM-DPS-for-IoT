'''
This script is the implementation of LSTM model descried in 
    @article{jarwan2019data,
      title={Data transmission reduction schemes in WSNs for efficient IoT systems},
      author={Jarwan, Abdallah and Sabbah, Ayman and Ibnkahla, Mohamed},
      journal={IEEE Journal on Selected Areas in Communications},
      volume={37},
      number={6},
      pages={1307--1324},
      year={2019},
      publisher={IEEE}
    }

The implementation is done by following the model describtion illustrated in the original authors' paper (as it is not available by the original authors), we 
tried to implement the model as best as we could. Of note, we tried to contact the authors to get the original model script by we got no reply.    
'''


from utils._model import Custom_Model, sort_tuble, get_data_look_back
from utils._experiment import Experiment, load_experiment
from utils._run import run_experiment,run_exp
from utils._experiment_setup import setup_
from utils._time_utils import get_timestamp_m, get_TimeStamp_str
from utils._utils import read_Intel_data, get_sorted_tuble_lst, load_mote_data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dropout, Dense, LSTM
from keras.models import Model, load_model
from keras import backend as K
from time import time
import numpy as np

import gc

tt = ''
emax_lst = [ 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 ]
for emax in emax_lst:

    # getting the best number of epochs
    for ep in [1,10,50,100][1:2]:
    
        # we run the experiment to get the mean and std of the experiments
        for repeats in range(5):
            input_layer = Input(shape=( 1, 10))
            out_model = LSTM(10, return_sequences=False)(input_layer)
            out_model = Dense(1, activation='sigmoid')(out_model)

            model = Model( inputs=[input_layer], outputs = [out_model] )
            model.compile( loss = 'mse', optimizer = 'sgd', metrics=['mae'] )
            def write_to_file(*str_):
                s = ''
                for arg in str_:
                    s += str(arg)
                    if str(arg) == '\n':pass
                    else:s = s  + ' '       
                s += '\n'
                with open('Ayman_res_%d_%-.1f.txt'%(ep,emax), 'a') as myfile:
                    myfile.write(s+'\n')
            df = read_Intel_data()
            data = load_mote_data( df, moteID = 30, date= ('2004-03-06', '2004-03-09') )
            s = MinMaxScaler()
            data['y'] = s.fit_transform(data['temperature'].values.reshape(-1,1))
            temp_scaled = data['y'].values
            miss = 11
            agg = temp_scaled[:11].tolist()

            total_time = 0.0
            for i in range(11, temp_scaled.shape[0]):
                print("\r", emax, ep, repeats, i,end="")
                inst = np.reshape(agg[-11:-1], (1,1, 10) )
                y = np.reshape(agg[-1],(1,1))
                a = time()
                model.fit( x=inst,
                           y=y, 
                           batch_size=1, 
                           epochs=ep, 
                           verbose=0 )
                write_to_file( 'inst = ' , inst , '    -> ',y)
                inst = np.reshape(agg[-10:], (1,1, 10) )
                pred = model.predict(inst)
                pred_inv = s.inverse_transform(pred)[0][0] 
                diff = np.abs( pred_inv- data['temperature'].values[i] )
                total_time += (time()-a)
                write_to_file( 'pred: (%-.3f, %-.3f)'% (pred, pred_inv) , ' True_val: (%-.3f, %-.3f)'%(temp_scaled[i], data["temperature"].values[i]), '   diff:',diff,'\n')
                if diff > emax:
                    agg.append( temp_scaled[i]  )
                    miss += 1
                else:
                    agg.append(pred_inv)

            K.clear_session()
            del model
            gc.collect()

            s ='%d_%-.1f_%d.txt'%(ep,emax, repeats) + '  miss = ' + str(miss) + '  acc=' + str((temp_scaled.shape[0]-miss)/temp_scaled.shape[0]) + '  total_time: %-.5f'%(total_time/60.0) +'  avg_time:%-.5f'%(total_time/(temp_scaled.shape[0]-11))
            write_to_file( s )
            print(s)
            with open('Ayman_all.txt', 'a') as myfile:
                myfile.write('\n'+s)
