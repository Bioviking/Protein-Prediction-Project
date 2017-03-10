#!/usr/bin/env bash
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:09:14 2017

@author: ryno
"""
################################Pipeline##########################

cd ../data/textfile/cross_validated/

for filename in *.txt
do 
    echo $filename > file_list.txt
done