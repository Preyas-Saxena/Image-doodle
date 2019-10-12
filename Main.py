
# Do the import
import torch
import Model
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import Data_processing

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
rootdir='/data/leuven/329/vsc32927/polyvore/'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_list=transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,])

num_epochs=1
learning_rate=1e-3

#Output is a variable containing tuples of (image,pos_image) pairs
data=Data_processing.datapairs(datadir=rootdir,filename='train_no_dup.json',transforms=transform_list)

#Getting the positive image pairs in batches
train_loader=DataLoader(data,batch_size=64,shuffle=True)

#Initializing a vgg net model with an embedding of 20 size, using a pre-trained model but will update all the weights
vgg = Model.initialize_model("vgg", 128, feature_extract=False, use_pretrained=True)
vgg.to(device)

#Initializing the optimizer
optimizer=optim.Adam(vgg.parameters(),lr=learning_rate)

#Training:
Model.train_model(vgg, train_loader, optimizer=optimizer, epochs=num_epochs, device=device)

#Generating sample embedding for an image from training data:
sample=data.__getitem__(0)[0]
sample=sample.unsqueeze(0)
sample=sample.to(device=device)

print(vgg(sample))
print("Done and dusted")


