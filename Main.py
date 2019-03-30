

scriptpath = "/home/preyas/anaconda3/bin/python /home/preyas/PycharmProjects/Thesis/DataLoader.py"
import os
import sys

#Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))

#Make sure to remove these paths later on

# Do the import
import Model
import DataLoader
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_list=transforms.Compose([transforms.Scale(224),
                               transforms.CenterCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize,])

rootdir='/home/preyas/Desktop/Thesis/polyvore/'
imgdir='/home/preyas/Desktop/Thesis/images/'

data=DataLoader.datapairs(datadir=rootdir,filename='valid_no_dup.json',transforms=transform_list)
#Output is a variable containing tuples of (image,pos_image) pairs

vgg,_ =Model.initialize_model("vgg",200,feature_extract=False,use_pretrained=True)
#Initializing a vgg net model with an embedding of 200 size, using a pre-trained model but will update all the weights








