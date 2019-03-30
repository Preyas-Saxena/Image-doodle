
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

def loss_function(anchor_embeds, pos_embeds):

    dot_product=torch.dot(anchor_embeds,pos_embeds)
    sigmoid_dot=torch.sigmoid(dot_product)
    loss=torch.log(sigmoid_dot)

    return loss


#Defining a training function:

def train_model(model,num_epochs,dataloader,loss_criterion,optimizer):

    for i in range(num_epochs):
        for batch_idx,(anchors,positives) in enumerate(dataloader):

            anchor_fc=model(anchors)
            pos_fc=model(positives)

            # loss=loss_criterion(anchor_fc,pos_fc)
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

    return model


def train_modelcopy(model,dataloader):

    for batch_idx,(anchors,positives) in enumerate(dataloader):
        anchor_fc=model(anchors)
        pos_fc=model(positives)

    return anchor_fc, pos_fc
