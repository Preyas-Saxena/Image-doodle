
scriptpath = "/home/preyas/anaconda3/bin/python /home/preyas/PycharmProjects/Thesis/Data_processing.py"
import os
import sys

#Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))

#Make sure to remove these paths later on

# Do the import
import Model
import Data_processing
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

rootdir='/home/preyas/Desktop/Thesis/polyvore/'
imgdir='/home/preyas/Desktop/Thesis/images/'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_list=transforms.Compose([transforms.Scale(224),
                               transforms.CenterCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize,])

imgs_batch=4

#Output is a variable containing tuples of (image,pos_image) pairs
data=Data_processing.datapairs(datadir=rootdir,filename='valid_no_dup_trim.json',transforms=transform_list)

# print(len(data))
# print(data[1][0].shape)

#Getting the positive image pairs in batches
train_loader=DataLoader(data,batch_size=2,shuffle=True)

# print(len(train_loader))
#
# for idx, (img1,img2) in enumerate(train_loader):
#
#     if idx==0:
#         print(img1.shape)#torch.Size([100,3,224,224])
#         print(img2.shape)#torch.Size([100,3,224,224])
#
#     else:
#         pass


#Initializing a vgg net model with an embedding of 20 size, using a pre-trained model but will update all the weights
vgg =Model.initialize_model("vgg",20,feature_extract=False,use_pretrained=True)


#Generating the embeddings from vgg model:
anchor_embed,pos_embed = Model.train_modelcopy(vgg,train_loader)

print(anchor_embed)
print("Shape of anchor embed:",anchor_embed.shape)












