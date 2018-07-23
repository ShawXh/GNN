import random
import json
#from config import *

def repairDict(d):
    for i in d:
        d[int(i)] = d.pop(i)
    return d

def checkDataset(ds):
    '''
        check over.
        The dataset is right.
    '''
    d = ds.raw_dict
    f = open("../check.log", "w")
    for i in d:
        for j in d[i]:
            if j not in d:
                f.write("[%d:%d] not in dict.\n" % (i,j))
                continue
            if i not in d[j]:
                f.write("[%d:%d] doesn\'t have reverse object in dict\n" \
                        % (i,j))
    f.close()

def getMaxIndex(d):
    max_ = 0
    for i in d:
        if i > max_:
            max_ = i
    return max_

class netDataset(object):
    def __init__(self, net_path, shuffle=False):
        f = open(net_path, 'r')
        self.raw_dict = repairDict(json.load(f))
        f.close()
        self.shuffle = shuffle
        if (self.shuffle):
            self.ordered_seq = range(1, getMaxIndex(self.raw_dict)+1)
            random.shuffle(self.ordered_seq)
            self.iter_obj = iter(self.ordered_seq)
        else:
            self.iter_obj = iter(self.raw_dict)

    def randomWalk(self, start_node, window_length=5):
        '''
            int start_node
            return:
                list path.
                path = [start_node, a, b, ...] 
                shape: [window_length + 1, ] 
        '''
#    if start_node not in self.raw_dict:
#        raise ValueError("The node %d is not in the net dict" % start_node)
        path = [start_node]
        current_node = start_node
#        print("Walk:\n\t%d" % current_node)
        for _ in xrange(window_length):
#            print("true" if current_node in self.raw_dict else "false")
            next_node = random.sample(self.raw_dict[current_node], 1)[0]
#            print("\t%d" % next_node)
            path.append(next_node)
            current_node = next_node
        return path

    def negativeSample(self, node, sample_num=35, negative_level=2):
        '''
            negative_level = 1: only the node will be assumed as the positive_node.
            negative_level = 2: the node and the direct adjascent node will be chose.
            return:
                list neg.
                shape: [sample_num, ]
        '''
        positive_nodes = []
        source_nodes = [node]
# render the positive nodes level by level
        while(True):
            tmp = []
            for p in source_nodes:
                if p not in positive_nodes:
                    tmp += self.raw_dict[p]
            positive_nodes += source_nodes
            source_nodes = tmp
            negative_level -= 1
            if negative_level <= 0:
                break
#        print("negative sampling.. positive nodes:",positive_nodes)
#        raise ValueError
# sampling for negative nodes
        neg = []
        i = 0
        while(i<sample_num):
            sample = random.sample(self.raw_dict, 1)[0]
            if sample not in positive_nodes:
                i += 1
                neg.append(sample)
        return neg

    def getBatchData(self, batch_size=16, window_length=5, negative_sample_num=35, negative_level=2, shuffle=False):
        '''
            get data for one batch
            return: batch_data, epoch_over
                batch_data: [batch_size, nums_nodes]
                e.g. [[start_node, p1, p2, ..., pi, n1, n2, ..., nj],
                      [start_node, p1, p2, ...                     ],
                      [...                                      ...]].
        '''
        i = 0;
        batch_data = []
        epoch_over = False
        while(i<batch_size):
# One epoch is over
            try:
                node = self.iter_obj.next()
#                print("node", node)
            except:
                if shuffle==True:
                    random.shuffle(self.ordered_seq)
                    self.iter_obj = iter(self.ordered_seq)
                else:
                    self.iter_obj = iter(self.raw_dict)
                epoch_over = True
                node = self.iter_obj.next()
# Walk the following node
            try:
                tmp = self.randomWalk(start_node=node, 
                                      window_length=window_length)
#                print("tmp", tmp)
                tmp += self.negativeSample(node=node,
                                           sample_num=negative_sample_num,
                                           negative_level=negative_level)
#                print("tmp", tmp)
                batch_data.append(tmp)
                i += 1
            except:
                continue
        return batch_data, epoch_over
