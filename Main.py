

scriptpath = "/home/preyas/anaconda3/bin/python /home/preyas/PycharmProjects/Thesis/Data_processing.py"
import os
import sys

#Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))

# Do the import
import Model
import Data_processing
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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
num_epochs=1
learning_rate=1e-5

#Output is a variable containing tuples of (image,pos_image) pairs
data=Data_processing.datapairs(datadir=rootdir,filename='valid_no_dup_trim.json',transforms=transform_list)
# data=Data_processing.datapairs_copy(filename='valid_no_dup_trim.json',transforms=transform_list)

#Getting the positive image pairs in batches
train_loader=DataLoader(data,batch_size=3,shuffle=True)

# for i,(a,b,c) in enumerate(train_loader):
#     print (a.shape)
#     break

#Initializing a vgg net model with an embedding of 20 size, using a pre-trained model but will update all the weights
vgg =Model.initialize_model("vgg",20,feature_extract=False,use_pretrained=True)

#Initializing the optimizer
optimizer=optim.Adam(vgg.parameters(),lr=learning_rate)

#Generating the embeddings from vgg model:
# anchor_embed,pos_embed,neg_embed1 = Model.train_modelcopy(vgg,train_loader,optimizer=optimizer,epochs=num_epochs)
anchor_embed,pos_embed = Model.train_model(vgg,train_loader,optimizer=optimizer,epochs=num_epochs)

print(anchor_embed.shape)



