# -*- coding: utf-8 -*-
"""
@author: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>
@brief: writting the output to file
"""

def write_to_file(*str_):
    s = ''
    for arg in str_:
        s += str(arg)
        if str(arg) == '\n':pass
        else:s = s  + ' '
        s += '\n'
    s += '\n\n'
    with open('res.txt', 'a') as myfile:
        myfile.write(s+'\n')

def dump(str, file_name='write_to_file_dump.txt'):
    with open(file_name, 'a') as myfile:
        myfile.write(str)