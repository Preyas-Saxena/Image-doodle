
import torch
import torch.nn.functional as F
import math
import torchvision.models as models
import torch.nn as nn
from datetime import datetime
from PIL import Image

# imgdir = '/home/preyas/Desktop/Thesis/images/'
imgdir='/data/leuven/329/vsc32927/images/'

# Function for setting the model parameters' requires_grad flag:
def set_params_requires_grad(model, feature_extracting):

    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Function for initializing the model as per the embedding size required and if extracting features versus training using the initial weights:
def initialize_model(model_name, embedding_size, feature_extract, use_pretrained):

    if model_name == "vgg":
        model = models.vgg16_bn(use_pretrained)
        set_params_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, embedding_size)

    else:
        pass

    return model

#Function for saving model params at various checkpoints throughout training:
def save_model(epoch, model_state_params, optimizer_state_param, scheduler_state_param, loss):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
    path_remote = '/data/leuven/329/vsc32927/Thesis_style/redo/' + dt_string + '_base_75gamma.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_params,
        'optimizer_state_dict': optimizer_state_param,
        'scheduler_state_dict': scheduler_state_param  ,
        'loss': loss}, path_remote)
    return None

#Function for saving model params at various checkpoints when validating:
def save_valid_model(epoch, model_state_params, optimizer_state_param, loss):
    path_remote = '/data/leuven/329/vsc32927/Thesis_style/valid.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_params,
        'optimizer_state_dict': optimizer_state_param,
        'loss': loss}, path_remote)
    return None

#Function for checking if the early stopping criteria is met (val loss has not decreased since last 3 checks):
def early_stopcheck(list, delta):
    check=False
    if ((list[-1]+ delta)> list[-2]) and ((list[-1]+ delta)> list[-3]) and ((list[-1]+ delta)> list[-4]):
        check=True
    return check

# Saves a ckpt after validation only if the current val loss is less than the previous one (also checks for early stopping):
def validation_ckpt(val_loss_list, current_val_loss, epoch, model_state_dict, optim_state_dict, train_loss):
    early_stopping=False
    delta = 5e-4

    if len(val_loss_list) > 1:

        if len(val_loss_list)>3:
            print('performing early stop check')
            early_stopping=early_stopcheck(val_loss_list, delta)

        if (current_val_loss + delta < val_loss_list[-2]):
            print('saving model because current val loss is less than previous one')
            save_valid_model(epoch=epoch, model_state_params=model_state_dict, optimizer_state_param=optim_state_dict, loss=train_loss)

        else:
            pass

    else:
        print('saving model after val for first time')
        save_valid_model(epoch=epoch, model_state_params=model_state_dict, optimizer_state_param=optim_state_dict, loss=train_loss)

    return early_stopping

#Defining a default image loader given the image's path:
def load_train_items(path, transform):
    img= Image.open(path).convert('RGB')
    if transform is not None:
        return transform(img)
    else:
        return img

#Function for taking a batch of outfit_ids, and their respective number of items and converting into a batch of tensors:
def outfit_to_tensor(outfits, outfit_len, transforms):
    # getting batch size:
    batch_size = outfits.shape[0]

    # Initializing list to stack items from each outfit in the current batch:
    item_list = []
    item_label_list = []

    for idx in range(batch_size):
        outfit_id = outfits[idx].item()
        num_items = outfit_len[idx].item()

        for item in range(1, num_items + 1):
            img_path = imgdir + str(outfit_id) + '/' + str(item) + ".jpg"
            img_tensor = load_train_items(img_path, transform=transforms)
            item_list.append(img_tensor)
            item_label_list.append(torch.tensor(outfit_id))

    # Stacking items from item_list in the form of a torch tensor- which can be used as an input to DL models:
    stacked_items = torch.stack(item_list)
    stacked_item_labels = torch.stack(item_label_list)

    return stacked_items, stacked_item_labels

#Function for computing distances matrix, given the input embeddings output from the model:
def distances(embeddings_matrix):

    embeddings_norm=F.normalize(embeddings_matrix, dim=1)
    dot_product = torch.matmul(embeddings_norm, torch.transpose(embeddings_norm, 0, 1))  # (3,3)
    sq_norm = torch.diag(dot_product, diagonal=0)  # (3,1)
    distance_matrix = torch.unsqueeze(sq_norm, 0) - 2 * dot_product + torch.unsqueeze(sq_norm, 1)
    distance_matrix = torch.clamp(distance_matrix, min=0.0)
    return distance_matrix

def get_triplet_mask(labels):

    indices_equal = torch.eye(labels.shape[0], dtype=torch.int16)
    indices_not_equal = torch.tensor(1) - indices_equal
    i_not_equal_j = torch.unsqueeze(indices_not_equal, dim=2)  # anchor and negative have the same indices
    i_not_equal_k = torch.unsqueeze(indices_not_equal, dim=1)  # anchor and positive have the same indices
    j_not_equal_k = torch.unsqueeze(indices_not_equal, dim=0)  # (anchor,negative), and (anchor,positive) are the same entity
    distinct_indices = torch.mul(torch.mul(i_not_equal_j, i_not_equal_k), j_not_equal_k)  # Just taking an overlap (or and) condition over all the three matrices above

    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)  # (5,5,1)
    i_equal_k = torch.unsqueeze(label_equal, 1)  # (5,1,5)
    not_i_equal_k = torch.tensor(1) - i_equal_k.type(torch.ShortTensor)
    valid_labels = torch.mul(i_equal_j.type(torch.ShortTensor), not_i_equal_k)

    mask = torch.mul(distinct_indices, valid_labels.type(torch.ShortTensor))
    return mask

def loss(embeds, labels, device):

    # Getting the squared-euclidean matrix:
    pairwise_dist = distances(embeds)

    # Calculating triplet loss for all possible combinations in the batch:
    anchor_pos_dist = torch.unsqueeze(pairwise_dist, 2)
    anchor_neg_dist = torch.unsqueeze(pairwise_dist, 1)

    margin = torch.tensor(0.2)
    triplet_loss = anchor_pos_dist - anchor_neg_dist + margin

    # Getting mask for identifying which pairs of triplets are valid:
    mask = get_triplet_mask(labels)
    triplet_loss = torch.mul(triplet_loss, mask.type(torch.FloatTensor).to(device))
    triplet_loss = torch.clamp(triplet_loss, min=0.0)  # Setting triplet loss to zero for easy negatives

    # Counting number of positive triplets:
    pos_triplets = torch.gt(input=triplet_loss, other=torch.tensor(1e-16).to(device))
    num_positive_triplets = torch.sum(pos_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_pos_triplets = torch.div(num_positive_triplets, (num_valid_triplets + torch.tensor(1e-16)))

    pos_triplet_loss = torch.sum(triplet_loss)/(num_positive_triplets + torch.tensor(1e-16))  # Adding a small value in the denominator to avoid NaN in case there is no positive triplet

    with torch.no_grad():
        overall_triplet_loss= torch.sum(triplet_loss)/(num_valid_triplets + torch.tensor(1e-16))
        a_p_pairs = torch.mul(mask.type(torch.FloatTensor).to(device), anchor_pos_dist)
        sum_ap_pairs = torch.sum(a_p_pairs)
        mean_ap_distance= sum_ap_pairs/torch.sum(pos_triplets)

        a_n_pairs = torch.mul(mask.type(torch.FloatTensor).to(device), anchor_neg_dist)
        sum_an_pairs = torch.sum(a_n_pairs)
        mean_an_distance= sum_an_pairs/torch.sum(pos_triplets)
        # print('Avg A-P dist: {:.2f}, Avg A-N dist: {:.2f}'.format(mean_ap_distance,mean_an_distance))

    return pos_triplet_loss, overall_triplet_loss, num_positive_triplets, num_valid_triplets, mean_ap_distance, mean_an_distance


def tnet(model, dataloader, valid_loader, optimizer, scheduler, epochs, transforms, device, validating):
    early_check=False
    validation_loss = []

    for epoch in range(1,epochs+1):
        if early_check:
            break

        batch_losses=[]
        batch_overall_losses=[]
        total_train_samples=0

        print('Current Learning rate is: ',optimizer.param_groups[0]['lr'],'and', scheduler.get_lr())

        for batch, (x, labels_x) in enumerate(dataloader):

            x, labels_x= outfit_to_tensor(x, labels_x, transforms=transforms)
            model.train()

            #Batch_embeds is the embeddings matrix
            batch_embeds = model(x.to(device))

            #Calculating mean length(norm) of embeddings:
            embeds_norm = torch.norm(batch_embeds, p=2, dim=1)
            mean_embeds_norm = torch.mean(embeds_norm)

            #Calculating triplet loss:
            triplet_loss, overall_triplet_loss, num_pos_triplets, num_valid_triplets, avg_ap_dist, avg_an_dist= loss(embeds=batch_embeds, labels=labels_x, device=device)

            #Breaking if loss becomes NaN for some reason:
            if math.isnan(triplet_loss):
                break

            model.zero_grad()  # makes grad equal to 0 to avoid accumulation of grads from last batch or iteration
            triplet_loss.backward()  # calculates grad of loss wrt to all the trainable parameters
            optimizer.step()  # Updates all the trainable parameters of the model

            #Storing batch loss:
            with torch.no_grad():

                batch_train_loss=triplet_loss.item()*num_pos_triplets.item()
                batch_train_overall_loss=overall_triplet_loss.item()*num_pos_triplets.item()

                batch_losses.append(batch_train_loss)
                batch_overall_losses.append(batch_train_overall_loss)

            total_train_samples=num_pos_triplets.item()+total_train_samples

            #Printing the batch loss after every iteration:
            if (((batch +1)%20) ==0):
                print('Epoch {}, Step {}/{}, Triplets: {}/{}, Loss: {:.2f}, Overall Loss: {:.2f}, A-P dist: {:.2f}, A-N dist: {:.2f}, Embeds Norm: {:.2f}'.format(epoch, (batch+1), len(dataloader) ,num_pos_triplets, num_valid_triplets, triplet_loss, overall_triplet_loss, avg_ap_dist, avg_an_dist, mean_embeds_norm))

            #Printing cumulative loss after every 100 iterations:
            if (batch+1)%400==0 or ((batch+1)==len(dataloader)):
                cum_loss= sum(batch_losses)/total_train_samples
                cum_overall_loss= sum(batch_overall_losses)/total_train_samples
                print('Epoch {}, After {} batches, Loss: {:.3f}, Overall Loss: {:.3f}'.format(epoch,(batch+1), cum_loss, cum_overall_loss))

            #Performing validation after every 20 percent of the training data:
            if ((batch+1)==800 or (batch+1)==len(dataloader)) and validating:
                print("Validation mode")
                with torch.no_grad():
                    val_batch_losses = []
                    val_batch_overall_losses = []
                    total_val_samples = 0

                    for val_batch, (val_x, val_labels_x) in enumerate(valid_loader):

                        val_x, val_labels_x = outfit_to_tensor(val_x, val_labels_x, transforms=transforms)
                        model.eval()
                        # Batch_embeds is the embeddings matrix
                        val_embeds = model(val_x.to(device))
                        # Calculating triplet loss:
                        val_triplet_loss, val_overall_triplet_loss, val_num_pos_triplets , _ , _ , _ = loss(embeds=val_embeds, labels=val_labels_x, device=device)

                        val_batch_losses.append(val_triplet_loss.item()*val_num_pos_triplets.item())
                        val_batch_overall_losses.append(val_overall_triplet_loss.item()*val_num_pos_triplets.item())
                        total_val_samples= val_num_pos_triplets.item() + total_val_samples

                    val_loss=sum(val_batch_losses)/total_val_samples
                    val_overall_loss=sum(val_batch_overall_losses)/total_val_samples
                    print('Validation Set, Loss:{:.3f}, Overall Loss:{:.3f}'.format(val_loss, val_overall_loss))

                    # validation_loss.append(val_overall_loss)
                    # early_check=validation_ckpt(val_loss_list=validation_loss, current_val_loss=val_overall_loss, epoch=epoch, model_state_dict=model.state_dict(), optim_state_dict=optimizer.state_dict(),train_loss=sum(batch_losses)/total_train_samples)
                    # if early_check:
                    #     break

            #Saving a checkpoint to monitor progress:
            if ((batch+1)%100)==0:
                now = datetime.now()
                dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
                bpath_remote='/data/leuven/329/vsc32927/Thesis_style/redo/ckpt/' + dt_string + '_epoch'+str(epoch) +'_batch'+ str(batch) +'.ckpt'
                torch.save({'batch_idx': batch}, bpath_remote)

            #Breaking after fixed batches/Testing:
            # if (batch+1)==100:
            #    break

        scheduler.step()
        save_model(epoch=epoch, model_state_params=model.state_dict(),optimizer_state_param=optimizer.state_dict(), scheduler_state_param=scheduler.state_dict(),loss=sum(batch_losses)/total_train_samples)
