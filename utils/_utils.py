# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: config file
"""

import datetime
import pandas as pd
import numpy as np

import traceback

import operator
def get_sorted_tuble_lst(tub_lst, reverse = True):
    return sorted(tub_lst, key=operator.itemgetter(1), reverse=reverse)

def read_Intel_data(plot = 0):
    def change_date(x):
        return str(x).replace('-','')
    df = pd.read_csv('data/Intel-DS.csv')
    df['date_modified'] = df['date'].apply(lambda x:change_date(x))
    df['time'] = df['time'].apply(lambda x:x[:8])
    df['time_stamp'] = df['date_modified'] + ':' + df['time']
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], format="%Y%m%d:%H:%M:%S")
    df.sort_values(by='time_stamp', inplace=True)
    return df

def read_Benzene_data():
    df = pd.read_csv('data/AirQualityUCI.csv', sep=';')
    dd = df[['C6H6(GT)']][:9357] # removing nan values   dd.dropna(inplace=True)
    dd = dd[~ (dd['C6H6(GT)']=='-200,0')] # removing '-200,0' values
    def to_float(x):
        return float(x.replace(',','.'))

    return dd['C6H6(GT)'].apply(lambda x:to_float(x)).values

def get_setup_data(setup_):
    update_policy = ''
    if 'employ_update_policy' in setup_.keys():update_policy=str(setup_['employ_update_policy'])
    s = 'Diff:'+str(setup_['Apply_diff'])+' '+ \
    ' Hist:'+str(setup_['train_on_historical_data'])+' '
    try:s+=    ' G_pr:'+str(setup_['initialization_period']+setup_['LMS_period'])+' '
    except:pass
    s+=    ' Updt:'+update_policy+'-'+str(setup_['block_size'])+'_'+str(setup_['block_err_threshold'])+'_'+str(setup_['phase_accumulated_block_err'])+' '+ \
    ' Dsze:'+str(setup_['data_size'])+' '+ \
    ' emax:'+str(setup_['e_max_threshold'])+' '+ \
    ' wind:%-2d/%-2d'%(setup_['LMS_w'],setup_['LSTM_w'])+' '+ \
    ' rati:'+str(setup_['train_validate_ratio'])+' '+ \
    ' rnft:'+str(setup_['number_random_fits'])+' '+ \
    ' hptr:'+str(setup_['hyperopt_max_trials'])+' '
    return s

def load_mote_data(data, moteID, date=None, attribute = 'temperature'):
    data = data[data.moteid == moteID][[attribute,'time_stamp','date']].copy() 
    data.sort_values(by='time_stamp', inplace=True)
    if date == None:
        return data[attribute]
    elif isinstance(date, tuple):
        return data[(data.date>= date[0]) & (data.date <= date[1])][attribute]
    elif isinstance(date, str):
        return data[(data.date==date)][attribute]