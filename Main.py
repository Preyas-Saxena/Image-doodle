
# Do the import
import torch
import Model
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import Data_processing

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
rootdir='/data/leuven/329/vsc32927/polyvore/'
train_file='valid_no_dup_trim.json'
valid_file='valid_no_dup_trim.json'

#Actual Values from the training dataset:
mean_data=[0.7170, 0.6794, 0.6613]
std_data=[0.2358, 0.2511, 0.2574]

normalize = transforms.Normalize(mean=mean_data, std=std_data)
transform_list=transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])

num_epochs=10
learning_rate=1e-4

#Output is a variable containing tuples of (image,pos_image) pairs
data=Data_processing.datapairs_updated(datadir=rootdir,filename=train_file,transforms=transform_list)
train_loader=DataLoader(data,batch_size=10,shuffle=False)

#Initializing a vgg net model with an embedding of 20 size, using a pre-trained model but will update all the weights
vgg = Model.initialize_model("vgg", 128, feature_extract=False, use_pretrained=True)
vgg.to(device)

#Initializing the optimizer
optimizer=optim.Adam(vgg.parameters(),lr=learning_rate)

#Training:
Model.train_model_oneneg(rootdir=rootdir, model=vgg ,dataloader=train_loader, optimizer=optimizer,epochs=num_epochs, device=device, validation_file=valid_file)
print("Done")

