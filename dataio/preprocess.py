import torch
import yaml
import pandas as pd
import numpy as np
from dataset import ctr_dateset,collate_fn
def read_config(config_file):

    pass

class Hash_Layer(torch.nn.Module):
    def __init__(self, hash_bin,batch_size,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hash_bin = hash_bin
        self.hash_func = hash
        self.batch_size = batch_size
        self.index =0
    
    def hash(self,x):
        return self.hash_func(x,self.hash_bin)

    def forward(self, x):
        numpy_x = np.array(x)
        self.index += 1
        # print(numpy_x,numpy_x.shape)
        y=[]
        # B,_ = numpy_x.shape
        for i in range(self.batch_size):
            y.append(self.hash_func(numpy_x[i])%self.hash_bin) 
        # return torch.tensor(y, dtype=torch.long)
        return numpy_x
        
        # print(hash_input)
        # return self.hash_func(x)%self.hash_bin

    
if __name__ == "__main__":
    datafile = '/home/zhengkang/work/work/ctr_rank/data/tiny_csv/train_sample.csv'
    data = pd.read_csv(datafile)
    configfile = '/home/zhengkang/work/work/ctr_rank/data/tiny_csv/config.csv'
    dataset = ctr_dateset(datafile,configfile)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)
    # for index,x in enumerate(dataloader):
    #     hash = Hash_Layer(10000,2)
    #     input = np.array(x['time_stamp'])
    #     print(input.shape)
    #     print(hash(input))
    input = torch.tensor(["aaaaa"])
    print(input)