
import torch
import torchvision.models as models
import torch.nn as nn

#Function for setting the model parameters' requires_grad flag:
def set_params_requires_grad(model,feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad=False

#Function for initializing the model as per the embedding size required and if extracting features versus training using the initial weights:

def initialize_model(model_name, embedding_size, feature_extract, use_pretrained=True):
    if model_name=="vgg":
        model = models.vgg16(use_pretrained)
        set_params_requires_grad(model,feature_extract)
        num_ftrs=model.classifier[6].in_features
        model.classifier[6]=nn.Linear(num_ftrs,embedding_size)
        # input_size = 224

    else:
        pass

    return  model

#Defining custom loss function for style2vec model:

def loss_functioncopy(anchor_embeds, pos_embeds, neg_embeds):

    neg_product=anchor_embeds.mul(neg_embeds*-1)
    dot_neg_product=torch.sum(neg_product,dim=1)
    sig_dot_neg=torch.sigmoid(dot_neg_product)
    log_sigmoid_neg=torch.log(sig_dot_neg)
    neg_loss=torch.sum(log_sigmoid_neg)

    element_product = anchor_embeds.mul(pos_embeds)
    dot_product = torch.sum(element_product, dim=1)
    sig_dot = torch.sigmoid(dot_product)
    log_sigmoid = torch.log(sig_dot)
    pos_loss = torch.sum(log_sigmoid)
    loss= pos_loss + neg_loss

    loss=loss*-1

    return loss


def loss_function(anchor_embeds, pos_embeds):

    # neg_product=anchor_embeds.mul(neg_embeds*-1)
    # dot_neg_product=torch.sum(neg_product,dim=1)
    # sig_dot_neg=torch.sigmoid(dot_neg_product)
    # log_sigmoid_neg=torch.log(sig_dot_neg)
    # neg_loss=torch.sum(log_sigmoid_neg)

    element_product = anchor_embeds.mul(pos_embeds)
    dot_product = torch.sum(element_product, dim=1)
    sig_dot = torch.sigmoid(dot_product)
    log_sigmoid = torch.log(sig_dot)
    pos_loss = torch.sum(log_sigmoid)
    # loss= pos_loss + neg_loss
    loss= pos_loss

    loss=loss*-1

    return loss


#Defining a training function:

def train_model(model,dataloader,optimizer,epochs):

    for i in range(epochs):
        for batch_idx,(anchors,positives) in enumerate(dataloader):
            anchor_fc=model(anchors)
            pos_fc=model(positives)

            loss=loss_function(anchor_fc,pos_fc)

            # if (batch_idx + 1) % 4 == 0:print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(i + 1, epochs, batch_idx + 1, len(dataloader), loss))

            model.zero_grad()
            loss.backward()
            optimizer.step()

    path='/home/preyas/PycharmProjects/Thesis/Test.ckpt'
    torch.save(model.state_dict(),path)

    return anchor_fc, pos_fc


def train_modelcopy(model,dataloader,optimizer,epochs):

    for i in range(epochs):
        for batch_idx,(anchors,positives,negative1) in enumerate(dataloader):
            anchor_fc=model(anchors)
            pos_fc=model(positives)
            neg_fc1=model(negative1)
            # neg_fc2=model(negative2)
            # neg_fc3=model(negative3)
            # loss=loss_function(anchor_fc,pos_fc,neg_fc)

            # if (batch_idx + 1) % 4 == 0:print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(i + 1, epochs, batch_idx + 1, len(dataloader), loss))
            # model.zero_grad()
            # loss.backward()
            # optimizer.step()


    return anchor_fc, pos_fc,neg_fc1






