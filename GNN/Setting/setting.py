# from util import *
import argparse
import pandas as pd
import numpy as np

"""
Require customers to input path of data, how many days they want to forecast and which city they focus
"""


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def set():
    #========================================================================
    # PARAM SETTING
    #========================================================================

    parser = argparse.ArgumentParser()

    parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--data',type=str,default='data_Printer_Shanghai/',help='data path') 

    parser.add_argument('--adj_data', type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--load_static_feature', type=str_to_bool, default=False,help='whether to load static feature')
    parser.add_argument('--cl', type=str_to_bool, default=True,help='whether to do curriculum learning')

    parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
    parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
    parser.add_argument('--subgraph_size',type=int,default=5,help='k') 
    parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes') 
    parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

    parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
    parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
    parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
    parser.add_argument('--end_channels',type=int,default=128,help='end channels')


    parser.add_argument('--in_dim',type=int,default=60,help='inputs dimension') 
    parser.add_argument('--seq_in_len',type=int,default=7,help='input sequence length') 
    parser.add_argument('--seq_out_len',type=int,default=7,help='output sequence length')

    parser.add_argument('--layers',type=int,default=3,help='number of layers')
    parser.add_argument('--batch_size',type=int,default=1,help='batch size')
    parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
    parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
    parser.add_argument('--clip',type=int,default=5,help='clip')
    parser.add_argument('--step_size1',type=int,default=2500,help='step_size')
    parser.add_argument('--step_size2',type=int,default=100,help='step_size')


    parser.add_argument('--epochs',type=int,default=250,help='') 
    parser.add_argument('--print_every',type=int,default=100,help='')
    parser.add_argument('--seed',type=int,default=101,help='random seed')
    parser.add_argument('--save',type=str,default='./GNN/save_GNN_model/',help='save path')
    parser.add_argument('--expid',type=int,default=1,help='experiment id')

    parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
    parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')

    parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')

    parser.add_argument('--runs',type=int,default=1,help='number of runs') 

    parser.add_argument('--val_ratio',type=int,default=118/993,help='ratio of validation set')
    parser.add_argument('--testSetRatio',type=float,default=254/1247,help='ratio of test set')

    parser.add_argument('--method',type=str,default="LR",help='model')

    parser.add_argument('--pre',type=str,default=False,help='Whether to load trained GNN model or not')
    parser.add_argument('--comp',type=str,default='cpu',help='use cpu or gpu')

    args = parser.parse_args(args=[])

    return args