from net_dataset import *


    
if __name__=="__main__":    
    ds = netDataset(net_path="../tt-fb/" + "twitter_network" + ".json")
    print("true" if 28034 in ds.raw_dict else "false")
#    checkDataset(ds)
