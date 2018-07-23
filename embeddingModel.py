import random
import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import json
import argparse
from net_dataset import *
import time
from config import *

def log_(string, file_name):
    f = open(file_name, 'a')
    f.write(string + '\n')
    f.close()

def flushLog(file_name):
    f = open(file_name, 'w')
    f.close()

class embeddingModel(nn.Module):
    def __init__(self, net_dict, log_path, embedding_dimension=50):
        super(embeddingModel, self).__init__()
        flushLog(log_path)
        self.dict_length = len(net_dict)
        self.embedding_dim = embedding_dimension
        self.log_path = log_path
        self.embedding_h = nn.Embedding(num_embeddings=self.dict_length,
                                        embedding_dim=embedding_dimension)
        self.embedding_u = nn.Embedding(num_embeddings=self.dict_length,
                                        embedding_dim=embedding_dimension)
    
    def log_(self, string):
        f = open(self.log_path, 'a')
        f.write(string + '\n')
        f.close()

    def forward(self, x, window_length):
        '''
        x: torch.LongTensor
        '''
        num_nodes, num_path_length = x.data.shape
#        self.log_("Input batch:" + str(x.data))
        p_ = self.calp(x)
        loss = -1 * t.log(t.cat((p_[:,:window_length], 1 - p_[:,window_length:]), dim=1))
        loss = loss.sum()
        #loss /= num_nodes
        self.log_("loss:" + str(loss))
        time.sleep(10)
        #loss = t.log(loss).sum()
        return loss

    def calp(self, x):
        num_nodes, num_path_length = x.data.shape
        self.log_("Input bacth:" + str(x))
        label_nodes = x[:,0]
        data_nodes = x[:,1:]
        u_ = self.embedding_u(data_nodes)
        # u_: [num_nodes, num_nodes_on_path, emvedding_dim]
        h_ = self.embedding_h(label_nodes)
        h_ = h_.view(num_nodes, self.embedding_dim, 1).contiguous()
        # h_: [num_nodes, embedding_dim, 1]
        p_ = u_.matmul(h_)
        # p_: [num_nodes, num_nodes_on_path, 1]
        p_ = p_.squeeze().contiguous()
        p_ = F.softmax(p_)
        return p_
    
    def dump(self, path):
        try:
            t.save(self.state_dict(), path)
        except:
            raise RuntimeError("Failed while saving model.")

    def load(self, path):
        try:
            self.load_state_dict(t.load(path))
        except:
            raise RuntimeError("Failed while loading model.")

if __name__=="__main__":
# initializing arguments
    parser = argparse.ArgumentParser(description="Train the Embedding Model")
    parser.add_argument("-n", "--net_name", choices=["tt", "fb"])
    parser.add_argument("-d", "--embedding_dimension", type=int, default=100)
    parser.add_argument("-l", "--lr", type=float, default=0.05)
    parser.add_argument("-e", "--epoch", type=int, default=200)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-w", "--window_length", type=int, default=3)
    parser.add_argument("--negative_level", type=int, default=3)
    parser.add_argument("--negative_sample_num" ,type=int, default=35)
    args = parser.parse_args()
    
    if args.net_name == "tt":
        net_name = "twitter_network"
        net_path = JSON_NEW_PATH['twitter-facebook']['tnet']
        log_path = LOG_PATH['twitter-facebook']['tnet']
    elif args.net_name == 'fb':
        net_name = "facebook_network"
        net_path = JSON_NEW_PATH['twitter-facebook']['fnet']
        log_path = LOG_PATH['twitter-facebook']['fnet']
    else:
        raise ValueError("net name is wrong.")
# initializing models
    ds = netDataset(net_path=net_path,
                    shuffle=False)
    MODEL = embeddingModel(net_dict = ds.raw_dict, 
                           log_path = log_path,
                           embedding_dimension = args.embedding_dimension)
    optimizer = t.optim.SGD(MODEL.parameters(), lr=args.lr)
    
    print("The embedding model:{\n \
           embedding dimension:%d\n \
           batch size:%d\n \
           window length:%d\n \
           negative sample num:%d}" % \
           (args.embedding_dimension,
            args.batch_size,
            args.window_length,
            args.negative_sample_num))
# training
    epoch = 0
    while(True):
        data = ds.getBatchData(batch_size= args.batch_size,
                               window_length= args.window_length, 
                               negative_sample_num= args.negative_sample_num,
                               negative_level= args.negative_level)
        if(data[1]==True):
            print('epoch:%d loss:%f' % (epoch, loss.data[0]))
            MODEL.dump("../tt-fb/" + net_path + "_epoch_%d.pkl" % epoch)
            epoch+=1
        data = V(t.LongTensor(data[0]))
        optimizer.zero_grad()
        loss = MODEL.forward(data, window_length=args.window_length)
        loss.backward()
        optimizer.step()
