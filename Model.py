
import torch
import Data_processing
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import math

from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader

#Actual Values from the training dataset:
mean_data=[0.7170, 0.6794, 0.6613]
std_data=[0.2358, 0.2511, 0.2574]

normalize = transforms.Normalize(mean=mean_data, std=std_data)
transform_list=transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])

#Function for loading validations set, ideally should be in Main. Have put here so that it can be called only when required i.e. post patience, so we can avoid loading this data when calling training :
def validation_loader(path, file):
    valid_data = Data_processing.datapairs_updated(datadir=path, filename=file,transforms=transform_list)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
    return valid_loader



# Function for setting the model parameters' requires_grad flag:
def set_params_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Function for initializing the model as per the embedding size required and if extracting features versus training using the initial weights:

def initialize_model(model_name, embedding_size, feature_extract, use_pretrained=True):
    if model_name == "vgg":
        model = models.vgg16(use_pretrained)
        set_params_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, embedding_size)
    else:
        pass
    return model


#Function which controls early stopping during training the model, it stops the training if the valid loss increases after a fixed number of initial epochs:

def early_stopping(current_epoch, valid_list, patience, delta=1e-1):
    if len(valid_list)>1:
        if current_epoch>patience:
            if valid_list[-1]> valid_list[-2]+delta:
                return False
            else:
                return True
        else:
            return True
    else:
        return True

#Defining custom loss function for style2vec model:
def loss_function_oneneg(anchor_embeds, pos_embeds, neg_embeds1):
    epsilon = 1e-5

    #Normalizing the embeds to unit length vectors:
    # anchor_pointsquare = torch.pow(anchor_embeds, 2)
    # anchor_sum = torch.sum(anchor_pointsquare, dim=1)
    # anchor_len = torch.pow(anchor_sum, 0.5)
    # anchor_len = torch.unsqueeze(anchor_len, 1)
    # anchor_embeds_norm = torch.div(anchor_embeds, anchor_len)
    #
    # pos_pointsquare = torch.pow(pos_embeds, 2)
    # pos_sum = torch.sum(pos_pointsquare, dim=1)
    # pos_len = torch.pow(pos_sum, 0.5)
    # pos_len = torch.unsqueeze(pos_len, 1)
    # pos_embeds_norm = torch.div(pos_embeds, pos_len)

    anchor_embeds_norm= F.normalize(anchor_embeds, dim=1)
    pos_embeds_norm= F.normalize(pos_embeds, dim=1)
    # neg_embeds1_norm= F.normalize(neg_embeds1, dim=1)

    # neg_product1=anchor_embeds.mul(neg_embeds1*-1)
    # dot_neg_product1=torch.sum(neg_product1,dim=1)
    # sig_dot_neg1=torch.sigmoid(dot_neg_product1)
    # sig_neg1_epsilon = sig_dot_neg1 + epsilon #Adding a small epsilon to make sure log doesnt take 0 as an input value
    # log_sigmoid_neg1=torch.log(sig_neg1_epsilon)

    element_product = anchor_embeds_norm.mul(pos_embeds_norm)
    dot_product = torch.sum(element_product, dim=1)
    sig_dot = torch.sigmoid(dot_product)
    sig_dotepsilon=sig_dot+epsilon #Adding a small epsilon to make sure log doesnt take 0 as an input value
    log_sigmoid = torch.log(sig_dotepsilon)

    #cumloss= log_sigmoid + log_sigmoid_neg1
    #loss= torch.mean(cumloss)

    loss=torch.mean(log_sigmoid) #Taking the mean value over all the pairs in this batch
    loss=loss*-1
    return loss


# Defining a training function with early stopping-not checked and integrated:
def train_model_oneneg(rootdir, model, dataloader, optimizer, epochs, device, validation_file, patience=20):
    valid_losses = []
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_samples = 0
        batch_train_losses = []
        for batch_idx, (anchors, positives, negative1) in enumerate(dataloader):
            batch_train_samples=anchors.size(0)
            anchor_fc = model(anchors.to(device))
            pos_fc = model(positives.to(device))
            neg_fc1 = model(negative1.to(device))
            train_loss = loss_function_oneneg(anchor_fc, pos_fc, neg_fc1)

            model.zero_grad()  # makes grad equal to 0 to avoid accumulation of grads from last batch or iteration
            train_loss.backward()  # calculates grad of loss wrt to all the trainable parameters
            optimizer.step()  # makes updates to all the parameters using grads calculated in the previous step

            if ((batch_idx + 1) % 1) == 0:
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, epochs, batch_idx + 1, len(dataloader), train_loss))

            with torch.no_grad():
                batch_train_loss=train_loss.item()*batch_train_samples
                batch_train_losses.append(batch_train_loss)
            total_train_samples=batch_train_samples+total_train_samples

            if ((batch_idx+1)%100)==0:
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
                bpath_remote='/data/leuven/329/vsc32927/Thesis_code/ckpt_test/' + dt_string + '_epoch'+str(epoch+1) +'_batch'+ str(batch_idx) +'.ckpt'
                torch.save({'batch_idx': batch_idx}, bpath_remote)

            if math.isnan(train_loss):
                break

        epoch_train_loss = sum(batch_train_losses) / total_train_samples
        train_losses.append(epoch_train_loss)
        torch.cuda.empty_cache()

        if (epoch+1)>(patience-1):#validate the model only if current epoch is more than patience, to save on computation and time
            print("Entering validation mode")
            model.eval()
            batch_valid_losses = []
            total_samples=0

            #Calling validation data loader:
            val_loader=validation_loader(path=rootdir,file=validation_file)

            with torch.no_grad():
                for batch_idx1, (anchors_valid, positives_valid, negative1_valid) in enumerate(
                        val_loader):
                    batch_samples=anchors_valid.size(0)
                    anchor_valid_fc = model(anchors_valid.to(device))
                    pos_valid_fc = model(positives_valid.to(device))
                    neg_fc1 = model(negative1_valid.to(device))
                    valid_loss = loss_function_oneneg(anchor_valid_fc, pos_valid_fc, neg_fc1)
                    valid_loss= valid_loss.item()*batch_samples
                    batch_valid_losses.append(valid_loss)  # appending the current batch loss to batch_valid_losses
                    total_samples=total_samples+batch_samples

                epoch_valid_loss= sum(batch_valid_losses)/total_samples
                valid_losses.append(epoch_valid_loss)  # appending epoch loss to the valid_losses list
                validation_check = early_stopping(epoch + 1, valid_losses, patience)
                print("Validation check is ", validation_check)
                print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss:{:.4f}'.format(epoch + 1, epochs, epoch_train_loss,
                                                                                    epoch_valid_loss))

                if validation_check == False: #Check for early stopping
                    print("Early stopping the training process")
                    break

                elif (epoch+1)%4==0 and ((epoch+1)!= epochs): #Saving a snapshot of model every 4 epochs
                    now = datetime.now()
                    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
                    path_remote = '/data/leuven/329/vsc32927/Thesis_code/' + dt_string + '_snap_Test.ckpt'
                    print("Saving a snapshot of model at ", epoch + 1, "epoch")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_train_loss}, path_remote)
                else:
                    pass

        elif (epoch+1)%5==0 and ((epoch+1)!= epochs):
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
            path_remote = '/data/leuven/329/vsc32927/Thesis_code/' + dt_string + '_snap_Test.ckpt'
            print("Saving a snapshot of model at ", epoch + 1, "epoch")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_train_loss}, path_remote)
        else:
            pass

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    path_remote = '/data/leuven/329/vsc32927/Thesis_code/' + dt_string + '_Test.ckpt'
    print("Saving the final model at ",epoch+1,"epoch")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_train_loss}, path_remote)

    return None


