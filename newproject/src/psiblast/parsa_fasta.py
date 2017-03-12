#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:58:49 2017

@author: ryno
"""

fname = open('../../data/fasta_files/id_seq_list.fasta', 'r+')
#fname = open('../../data/fasta_files/test_fasta/test_fasta.fasta', 'r+')
#out_test = open('../../data/fasta_files/test_fasta/xfile.fasta', 'w')

def parse_fasta(filename):
#    f = open(filename,'r')
    g_dict = dict()
    for line in filename:
        line = line.strip()
        #print(line)
        if line[0] == '>':
            line = line[0:]
            #print(line)
            temp_id = line
            out_test = open('../../data/fasta_files/fasta_samples/' + temp_id + '.fasta', 'w')
            #print(temp_id)
            out_test.write(line + '\n')
            
        else:
            g_dict[temp_id] = line
            out_test.write(line + '\n')
    out_test.close()
    return g_dict

parse_fasta(fname)
