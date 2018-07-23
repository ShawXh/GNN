# coding: utf-8

import csv
import json
from config import *

alignment_csv_path = CSV_PATH['twitter-facebook']['align']
fnet_csv_path      = CSV_PATH['twitter-facebook']['fnet']
tnet_csv_path      = CSV_PATH['twitter-facebook']['tnet']
alignment_json_path = JSON_PATH['twitter-facebook']['align']
fnet_json_path      = JSON_PATH['twitter-facebook']['fnet']
tnet_json_path      = JSON_PATH['twitter-facebook']['tnet']
fnet_json_new_path  = JSON_NEW_PATH['twitter-facebook']['fnet'] 
fnet_json_raw2new_path = JSON_NEW_PATH['twitter-facebook']['fnet_raw2new']
fnet_json_new2raw_path = JSON_NEW_PATH['twitter-facebook']['fnet_new2raw']
tnet_json_new_path  = JSON_NEW_PATH['twitter-facebook']['tnet']
tnet_json_raw2new_path = JSON_NEW_PATH['twitter-facebook']['tnet_raw2new']
tnet_json_new2raw_path = JSON_NEW_PATH['twitter-facebook']['tnet_new2raw']

def append_(net_dict, row_in_csv):
    '''
        row_in_csv [int, int]
        test over.
    '''
    a, b = row_in_csv
    try:
        net_dict[a].append(b)
    except:
        net_dict[a] = []
        net_dict[a].append(b)
    try:
        net_dict[b].append(a)
    except:
        net_dict[b] = []
        net_dict[b].append(a)
    return True

def netIndexContinuous(net_raw):
    '''
        cuz' the index in the raw json dict is not continuous, so it's neccessary
        to make the index continuos and generate the new json file in the same
        time.
    '''
    net_new = {}
    map_raw2new = {}
    map_new2raw = {}
    i = 0
    for node in net_raw:
        if node not in net_new:
            map_raw2new[node] = i
            i += 1
    def map_func(raw):
        return map_raw2new[raw]
    for node in net_raw:
        new_node = map_func(node)
        net_new[new_node] = map(map_func, net_raw[node])
    for raw in map_raw2new:
        new = map_raw2new[raw]
        map_new2raw[new] = raw
    return net_new, map_raw2new, map_new2raw

def checkRaw2New(net_raw, net_new, map_raw2new):
    '''
        To check whether the newly generated json dict is right. I have checked 
        1->0, and it seems right. Three parameters above are all dict.
        net_raw: int 2 int
        net_new: int 2 int
        map_raw2new: str 2 int
    '''
    flag = True
    for node in net_raw:
        for i in net_raw[node]:
            if map_raw2new[str(i)] not in net_new[map_raw2new[str(node)]]:
                flag = False
    return flag

def net_csv2json(net_name):
    '''
        net_name = 'fb' or 'tt'
        test over.
    '''
    net_path = fnet_csv_path if net_name=='fb'\
               else tnet_csv_path if net_name=='tt'\
               else None
    json_path = fnet_json_path if net_name=='fb'\
                else tnet_json_path if net_name=='tt'\
                else None
    json_new_path = fnet_json_new_path if net_name=='fb'\
                    else tnet_json_new_path if net_name=='tt'\
                    else None
    json_raw2new_path = fnet_json_raw2new_path if net_name=='fb'\
                        else tnet_json_raw2new_path if net_name=='tt'\
                        else None
    json_new2raw_path = fnet_json_new2raw_path if net_name=='fb'\
                        else tnet_json_new2raw_path if net_name=='tt'\
                        else None
    if net_path==None or \
       json_path==None or \
       json_new_path==None or \
       json_raw2new_path==None or \
       json_new2raw_path==None:
        raise ValueError('net_name is wrong!')
# append net dataset.
    net = {}
    with open(net_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            break
        for row in reader:
            row_ = map(int, row)
            append_(net, row_)
    with open(json_path, 'w') as f:
        json.dump(net, f)
# make the index continuous
    net_new, map_raw2new, map_new2raw = netIndexContinuous(net)
    with open(json_new_path, 'w') as f:
        json.dump(net_new, f)
    with open(json_raw2new_path, 'w') as f:
        json.dump(map_raw2new, f)
    with open(json_new2raw_path, 'w') as f:
        json.dump(map_new2raw, f)
    return True

def net_alignment2json():
    align = {}
    with open(alignment_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row_ = map(int, row)
            try:
                align[row_[0]].append(row_[1])
            except:
                align[row_[0]] = []
                align[row_[0]].append(row_[1])
    with open(alignment_json_path, 'w') as f:
        json.dump(align, f)

if __name__=="__main__":
    net_csv2json('fb')
    print('facebook net csv2json finished.')
    net_csv2json('tt')
    print('twitter net csv2json finished.')
#   net_alignment2json()
#   print('alignment csv2json finished.')
