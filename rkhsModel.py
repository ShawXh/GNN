import random
import torch as t
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import json
import argparse
from net_dataset import *
from embeddingModel import *

class rkhsModel(nn.Module):
    '''
        map net2 space to net1.
        cuz' we assume net1 as the obejct RKHS
    '''        
    def __init__(self, net1_model_path, net1_dict, net2_model_path, net2_dict, embedding_dimension=50):
        '''
            embedding_net1/2 is a .pkl file
        '''
        super(rkhsModel, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.net_1 = embeddingModel(net_dict=net1_dict,
                                    embedding_dimension=embedding_dimension)
        self.net_1.load_state_dict(t.load(net1_model_path))
        self.net_2 = embeddingModel(net_dict=net2_dict,
                                    embedding_dimension=embedding_dimension)
        self.net_2.load_state_dict(t.load(net2_model_path))
        self.map_layer = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
    
    def getEmbeddingSet(self, net_id):
        '''
            return the embedding of the whole dataset.
        '''
        if net_id==1:
            length = self.net_1.dict_length
            nodes = V(t.LongTensor(range(1,length)))
            embedding_u = self.net_1.embedding_u(nodes)
            embedding_h = self.net_1.embedding_h(nodes)
            return (embedding_u, embedding_h)
        elif net_id==2:
            length = self.net_2.dict_length
            nodes = V(t.LongTensor(range(1,length)))
            embedding_u = self.net_2.embedding_u(nodes)
            embedding_h = self.net_2.embedding_h(nodes)
            return (embedding_u, embedding_h)
        else:
            raise ValueError("The given net_id %d is wrong!" % net_id)

    def calLoss(self, lmd=0.1):
        '''
            mat_set1/2: [num_nodes, embedding_dim] Tensor
        '''
        mat_set1_u, mat_set1_h = self.getEmbeddingSet(net_id=1)
        mat_set2_u, mat_set2_h = self.getEmbeddingSet(net_id=2)
        set1_u_mean = t.mean(mat_set1_u, dim=0) 
        set1_h_mean = t.mean(mat_set1_h, dim=0) 
        set2_u_mean = t.mean(mat_set2_u, dim=0) 
        set2_h_mean = t.mean(mat_set2_h, dim=0) 
        loss = t.pow(self.map_layer.parameters().next() - V(t.eye(self.embedding_dimension)), 2) + \
               lmd * (t.pow((set1_u_mean - set2_u_mean), 2) + \
               t.pow((set1_h_mean - set2_h_mean), 2))
        return loss.sum()

    def forward(self, lmd=0.1, batch_size=16):
        '''
        '''
        return self.calLoss(lmd)

    def dump(self, path):
        try:
            t.save(self.state_dict(), path)
        except:
            raise RuntimeError("Failed while saving model.")

def rkhsTrain(model, num_epochs, lr, lmd):
    optimizer = t.optim.SGD(model.parameters(), lr=lr)
    for epoch in xrange(num_epochs):
        optimizer.zero_grad()
        loss = model.forward(lmd=lmd)
        if epoch%5==0:
            print("Epoch:%d, loss:%f" % (epoch, loss.data[0]))
        loss.backward()
        optimizer.step()
    model.dump("rkhsModel.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train for graph macthing rkhsModel \
                                     and also contains code for the model.")
    parser.add_argument("-e", "--num_epochs", type=int, default=50)
    parser.add_argument("-l", "--lr", type=float, default=0.05)
    parser.add_argument("--lmd", type=float, default=0.1)
    args = parser.parse_args()

    ds1 = netDataset(net_path='net1.json')
    ds2 = netDataset(net_path='net2.json')

    model = rkhsModel(net1_model_path='net1.pkl',
                      net1_dict=ds1.raw_dict,
                      net2_model_path='net2.pkl',
                      net2_dict=ds2.raw_dict,
                      embedding_dimension=50)
    rkhsTrain(model,
              num_epochs = args.num_epochs,
              lr = args.lr,
              lmd = args.lmd)
    print("Train over.")
