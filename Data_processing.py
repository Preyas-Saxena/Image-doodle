
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
# import torch.nn.functional as F
# import torchvision
# from torchvision import transforms
# import matplotlib.pyplot as plt

imgdir='/data/leuven/329/vsc32927/images/'

#Defining a default image loader given the image's path:
def default_image_loader(path):
    return Image.open(path).convert('RGB')

class datapairs(Dataset):

    def __init__(self,datadir,filename,transforms=None):
        self.outfit_data=json.load(open(str(datadir+filename),'r'))
        pos_pairs = []
        max_items = 0
        cat2ims={}

        for outfit in self.outfit_data:
            items = outfit['items']
            cnt = len(items)
            max_items = max(cnt, max_items)
            outfit_id = outfit['set_id']
            for j in range(cnt - 1):
                for k in range(j + 1, cnt):
                    pos_pairs.append([outfit_id, items[j]['index'], items[k]['index']])

        for outfit in self.outfit_data:
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

        return img1,img2


#Testing1

# rootdir='/home/preyas/Desktop/Thesis/polyvore/'
# from torchvision import transforms
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# transform_list=transforms.Compose([transforms.Scale(224),transforms.CenterCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
#
# data=datapairs(datadir=rootdir,filename='valid_no_dup_trim.json',transforms=transform_list)


class datapairs_copy(Dataset):

    def __init__(self,datadir,filename,transforms=None):

        self.outfit_data=json.load(open(str(datadir+filename),'r'))
        pos_pairs = []
        max_items = 0

        for outfit in self.outfit_data:
            items = outfit['items']
            cnt = len(items)
            max_items = max(cnt, max_items)
            outfit_id = outfit['set_id']
            for j in range(cnt - 1):
                for k in range(j + 1, cnt):
                    pos_pairs.append([outfit_id, items[j]['index'], items[k]['index']])

        im2type = {}
        category2ims = {}

        for outfit in self.outfit_data:
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['index']
                category = item['categoryid']

                if outfit_id not in im2type:
                    im2type[outfit_id] = {}

                if im not in im2type[str(outfit_id)]:
                    im2type[str(outfit_id)][im]=category

                if category not in category2ims:
                    category2ims[category] = {}

                if outfit_id not in category2ims[category]:
                    category2ims[category][outfit_id] = []

                category2ims[category][outfit_id].append(im)

        self.pos_pairs=pos_pairs
        self.category2ims=category2ims
        self.im2type=im2type
        self.transform=transforms

#Writing function for retrieving negative samples same as the item's type but from different style sets

    def negative_samples(self,outfit_id,anchor_index):
        anchor_catg=self.im2type[str(outfit_id)][anchor_index]
        candidate_sets=self.category2ims[anchor_catg]
        neg_samples=[]
        ctr = 2

        for key in candidate_sets.keys():
            if len(neg_samples)<ctr:
                if str(outfit_id)==key:
                    pass
                else:
                    neg_samples.append((key,candidate_sets[key][0]))
            else:
                break
        return neg_samples


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
        neg_list=self.negative_samples(outfit_id,anchor_im)

        len_negs= len(neg_list)

        neg_outfit1,neg_im1=neg_list[0]
        # neg_outfit2,neg_im2=neg_list[1]
        # neg_outfit3,neg_im3=neg_list[2]

        img1_path=imgdir+str(outfit_id)+'/'+str(anchor_im)+".jpg"
        img2_path=imgdir+str(outfit_id)+'/'+str(pos_im)+".jpg"

        img3_path=imgdir+neg_outfit1+'/'+str(neg_im1)+".jpg"
        # img4_path=imgdir+neg_outfit2+'/'+str(neg_im2)+".jpg"
        # img5_path=imgdir+neg_outfit3+'/'+str(neg_im3)+".jpg"

        img1 = self.load_train_items(img1_path)
        img2 = self.load_train_items(img2_path)

        img3 = self.load_train_items(img3_path)
        # img4 = self.load_train_items(img4_path)
        # img5 = self.load_train_items(img5_path)

        # return img1,img2, img3, img4
        return img1,img2, img3


