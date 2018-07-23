from net_dataset import *
from config import *
import json
import csv
import argparse

def randomWalk(json_net_path, csv_out_path, shuffle=False, **kargs):
    if "window_length" not in kargs:
        kargs["window_length"] = 2
    if "negative_sample_num" not in kargs:
        kargs["negative_sample_num"] = 37 
# initialize raw dataset
    ds = netDataset(json_net_path, shuffle=shuffle)
# initialize writer
    f = open(csv_out_path, 'wb')
    row_first = ["start_node"] + \
                ["positive"] * kargs["window_length"] + \
                ["negative"] * kargs["negative_sample_num"]
    csv_writer = csv.writer(f)
    csv_writer.writerow(row_first)
# begin write
    count = 0
    epoch_over = False
    while(not epoch_over):
        data, epoch_over = ds.getBatchData(batch_size = 1,
                               window_length = kargs["window_length"],
                               negative_sample_num= kargs["negative_sample_num"],
                               negative_level= kargs["window_length"])
        if epoch_over:
            break
        csv_writer.writerow(data)
        count += 1
        if count%500==0:
            print("\t%d passed." % count)
    f.close()
    return True

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate the csv dataset through random walking")
    parser.add_argument("-n", "--net_name", choices=["tt", "fb"])
    parser.add_argument("-w", "--window_length", type=int, default=3)
    parser.add_argument("--number", type=str, default='0', help="No. of Walking")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--negative_sample_num" ,type=int, default=26)
    args = parser.parse_args()

    if args.net_name == "tt":
        net_name = "twitter_network"
        net_path = JSON_NEW_PATH['twitter-facebook']['tnet']
        out_path = RANDOM_WALK_CSV_PATH['twitter-facebook']['tnet'][-4:] +\
                   "_" + args.number +\
                   ".csv"
    elif args.net_name == 'fb':
        net_name = "facebook_network"
        net_path = JSON_NEW_PATH['twitter-facebook']['fnet']
        out_path = RANDOM_WALK_CSV_PATH['twitter-facebook']['fnet'][-4:] +\
                   "_" + args.number +\
                   ".csv"
    
    print("Starting to walk...")
    randomWalk(json_net_path = net_path,
               csv_out_path = out_path,
               shuffle = args.shuffle,
               window_length = args.window_length,
               negative_sample_num = args.negative_sample_num)
    print("Walking over")
