# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <fathalla_sci@science.suez.edu.eg>
@brief: utils for time

"""
DayL = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

import datetime

def get_TimeStamp_str():
    now = datetime.datetime.now()
    h = int(now.strftime("%H"))
    m = int(now.strftime("%M"))
    ss = int(now.strftime("%S"))
    s = now.strftime("%Y-%m-%d")
    s = str(s)
    if m<10:
        m = '0' + str(m)
    if h>12:
        s += "*%2d_%2s_%2dPM"%(h - 12,m,ss)
    elif h == 12:
        s += "*%2d_%2s_%2dPM"%(h ,m,ss)
    else:
        s += "*%2d_%2s_%2dAM"%(h,m,ss)
    return DayL[datetime.datetime.now().weekday()]+'_'+s.replace(' ','0').replace('*',' ')
	
def get_timestamp_m(option_ = 1):
    now = datetime.datetime.now()
    h = int(now.strftime("%H"))
    m = int(now.strftime("%M"))
    ss = int(now.strftime("%S"))
    s = now.strftime("%Y/%m/%d")
    s = str(s)
    if m<10:
        m = '0' + str(m)
    if h>12:
        s += "*%2d:%2s:%2d*PM"%(h - 12,m,ss)
    elif h == 12:
        s += "*%2d:%2s:%2d*PM"%(h ,m,ss)
    else:
        s += "*%2d:%2s:%2d*AM"%(h,m,ss)

    if option_==0:return s.replace('/','_').replace(':','_').replace(' ','_')
    elif option_ == 1:return s.replace(' ','0').replace('*',' ')
    elif option_ == 2:return ' ***running at: '+ str(s)
