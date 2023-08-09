import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import default_collate

feature_colums = []
class ctr_dateset(torch.utils.data.Dataset):
    def __init__(self, csv_data_path,config_file_path):
        self.data_path = csv_data_path
        self.config_file = config_file_path
        self.data = pd.read_csv(self.data_path)
        self.clk_label = self.data['clk']
        self.data = self.data.drop(['clk'],axis=1)
    def __getitem__(self,index):
        ctr_label = self.clk_label[index]
        feature = {}
        for feature_name in self.data.columns:
            feature_colums.append(feature_name)
            feature[feature_name] = self.data[feature_name][index]
        return feature,ctr_label
    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    """
    动态padding,返回Tensor
    """
    feature,label = zip(*batch)   
    new_batch = {}
    for feature_name in feature_colums:
        feature_name_list = []
        for index in range(len(feature)):   
            feature_name_list.append(feature[index][feature_name])
        new_batch[feature_name] = feature_name_list
    new_batch['clk'] = label
    return new_batch

if __name__ =='__main__':
    datafile = '/home/zhengkang/work/work/ctr_rank/data/tiny_csv/train_sample.csv'
    data = pd.read_csv(datafile)
    configfile = '/home/zhengkang/work/work/ctr_rank/data/tiny_csv/config.csv'
    dataset = ctr_dateset(datafile,configfile)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)
    for x in enumerate(dataloader):
        print(x)