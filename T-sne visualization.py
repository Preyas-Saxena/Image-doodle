
import Model
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from sklearn.manifold import TSNE
import json

train_file='train_no_dup.json'
rootdir = '/home/preyas/Desktop/Thesis/polyvore/'
imgdir = '/home/preyas/Desktop/Thesis/images/'

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
# device='cpu'

#Actual Values from the training dataset:
mean_data=[0.7170, 0.6794, 0.6613]
std_data=[0.2358, 0.2511, 0.2574]

normalize = transforms.Normalize(mean=mean_data, std=std_data)
transform_list=transforms.Compose([transforms.Scale(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize,])

#Loading trained model for generating feature vectors:
vgg = Model.initialize_model("vgg", 128, feature_extract=False, use_pretrained=True)

#Loading weights from checkpoint- optional:
PATH='/home/preyas/PycharmProjects/Thesis_style/18-11-2019_14:25:08_base.ckpt'
checkpoint = torch.load(PATH, map_location= torch.device('cpu'))
vgg.load_state_dict(checkpoint['model_state_dict'])

vgg.to(device)
vgg.eval()

#Defining a default image loader given the image's path:
def load_train_items(path, transform):
    img= Image.open(path).convert('RGB')
    if transform is not None:
        return transform(img)
    else:
        return img

#Function which returns a list of image paths:
def img_paths(file, file_dir, img_dir):
    outfit_data = json.load(open(str(file_dir + file), 'r'))
    paths=[]
    for outfit in outfit_data:
        set_id=outfit['set_id']
        num_items=len(outfit['items'])

        for item in range(1, num_items+1):
            img_path= img_dir + str(set_id)+'/'+str(item)+'.jpg'
            paths.append(img_path)

    return paths

def catg_img_paths(file, file_dir, img_dir, catgs_list):
    outfit_data = json.load(open(str(file_dir + file), 'r'))
    paths = []

    for outfit in outfit_data:
        set_id = outfit['set_id']
        items=outfit['items']
        for item in items:
            if item['categoryid'] in catgs_list:
                img_path= img_dir + str(set_id)+'/'+str(item['index'])+'.jpg'
                paths.append(img_path)
    return paths

#Selecting only a few subset of all the images : their paths for visualization:
# images_path = imgdir
# image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)
# images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]

max_num_images = 1300
cats=[11,261,43,46]
# images = img_paths(file=train_file, file_dir= rootdir, img_dir=imgdir)
images = catg_img_paths(file=train_file, file_dir= rootdir, img_dir=imgdir, catgs_list=cats)


# if max_num_images < len(images):
#     images = [images[i] for i in sorted(random.sample(range(len(images)), max_num_images))]

if max_num_images<len(images):
    images= images[0:max_num_images]

print("keeping %d images to analyze" % len(images))

#Generating the feature vectors for the images selected above using the pretrained model:
features = []
for i, image_path in enumerate(images):
    img = load_train_items(image_path, transform=transform_list)
    with torch.no_grad():
        feat = vgg(torch.unsqueeze(img, dim=0).to(device))
        feat= torch.squeeze(feat)
    features.append(feat.cpu().numpy())

print('finished extracting features for %d images' % len(images))

#Performing t-sne on the features:
X=np.array(features)
tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

#Normalizing t-sne embeddings so that each axis value lies in the range 0-1:
tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

#Pasting full-images based on tx and ty values learned from t-sne:
width = 4000
height = 3000
max_dim = 150

full_image = Image.new('RGBA', (width, height))
for img, x, y in zip(images, tx, ty):
    tile = Image.open(img)
    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

plt.figure(figsize = (16,12))
plt.imshow(full_image)
