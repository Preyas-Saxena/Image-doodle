

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

rootdir='/home/preyas/Desktop/Thesis/polyvore/'
imgdir='/home/preyas/Desktop/Thesis/images/'

os.chdir(rootdir)
outfit_data = json.load(open('valid_no_dup.json', 'r'))
num_batches=10



#Defining a default image loader:

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class datapairs(Dataset):

    def __init__(self,datadir,filename,transforms=None):
        self.outfit_data=json.load(open(str(filename),'r'))
        pos_pairs = []
        max_items = 0
        cat2ims={}

        for outfit in outfit_data:
            items = outfit['items']
            cnt = len(items)
            max_items = max(cnt, max_items)
            outfit_id = outfit['set_id']
            for j in range(cnt - 1):
                for k in range(j + 1, cnt):
                    pos_pairs.append([outfit_id, items[j]['index'], items[k]['index']])

        for outfit in outfit_data:
            items = outfit['items']
            cnt = len(items)

            for i in range(cnt):
                new_name = outfit['set_id'] + '_' + str(items[i]['index'])
                cat2ims[str(new_name)] = items[i]['categoryid']

        self.pos_pairs=pos_pairs
        self.transform=transforms
        self.cat2ims=cat2ims

    def load_train_items(self,path):
        img=default_image_loader(path)
        if self.transform is not None:
            return self.transform(img)
        else:
            return img

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index):
        outfit_id, anchor_im, pos_im = self.pos_pairs[index]
        img1_path=imgdir+str(outfit_id)+'/'+str(anchor_im)+".jpg"
        img2_path=imgdir+str(outfit_id)+'/'+str(pos_im)+".jpg"

        img1 = self.load_train_items(img1_path)
        img2 = self.load_train_items(img2_path)

        return img1, img2


# data=datapairs(datadir=rootdir,filename='valid_no_dup.json',transforms=transform_list)

# print(type(data))
# print(len(data[24460]))
# print(data[1][0].shape) #torch.size([3,224,224])

# Img=data[1][1]
# plt.imshow(Img.permute(1, 2, 0))
# plt.show()

# train_loader=DataLoader(data,batch_size=num_batches,shuffle=True)









































