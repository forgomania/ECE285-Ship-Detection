#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import os
import torch
from inceptionresnetv2.pytorch_load import InceptionResnetV2
from pathlib import Path
import random
import cv2
from torchvision.transforms import ToTensor, Normalize, Compose


# In[2]:


"""
    Implementation from  https://github.com/ternaus/robot-surgery-segmentation
    Data augemntation methods
    
"""

class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask
    
class OneOf:
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            x, mask = t(x, mask)
        return x, mask

class OneOrOther:
    def __init__(self, first, second, prob=0.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

    def __call__(self, x, mask=None):
        if random.random() < self.prob:
            x, mask = self.first(x, mask)
        else:
            x, mask = self.second(x, mask)
        return x, mask


class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        return self.trans(x), mask


class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class Transpose:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2)
            if mask is not None:
                mask = mask.transpose(1, 0, 2)
        return img, mask


class RandomRotate90:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()


class Rotate:
    def __init__(self, limit=90, prob=0.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class RandomCrop:
    def __init__(self, size):
        self.h = size[0]
        self.w = size[1]

    def __call__(self, img, mask=None):
        height, width, _ = img.shape

        h_start = np.random.randint(0, height - self.h)
        w_start = np.random.randint(0, width - self.w)

        img = img[h_start: h_start + self.h, w_start: w_start + self.w,:]

        assert img.shape[0] == self.h
        assert img.shape[1] == self.w

        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=2)
            mask = mask[h_start: h_start + self.h, w_start: w_start + self.w,:]

        return img, mask

class RandomBrightness:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * img[..., :3], dtype, maxval)
        return img


# In[3]:


'''
set the environment and data file path
'''

USE_GPU = True

dtype = torch.float32 # we will be using float for most data
# the default is gpu but if gpu is not available,then use cpu
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# show which device to use
print('using device:', device)

ship_dir = '/datasets/ee285f-public/airbus_ship_detection/'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')


# In[4]:


'''
read this file to get the image file name and the label(0 or 1) and turn it into numpy
then seperate it into train and validation
'''

label = pd.read_csv('train_ship_label.csv')

np_la = label.values
np_train = np_la[0:180000,0]
np_valid= np_la[180000:192556,0]
total = len(np_train)
tlabel = np_la[0:180000,2]
vlabel = np_la[180000:192556,2]
print(total)


# In[5]:


'''
train_transform is to do some augmentation the the image
transform is to convert the data from (0 ,255) to (0,1)
'''

train_transform = DualCompose([
        HorizontalFlip(),
        VerticalFlip(),
        #ImageOnly(RandomBrightness()),
        #ImageOnly(RandomContrast()),
])
transform = Compose([
        ToTensor(),
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# In[7]:


'''
set the train parameter,flag and load the model(if exist)
'''


net = InceptionResnetV2().cuda()
# we use cross entropy to calculate the loss
criterion = nn.CrossEntropyLoss()

model_path = Path('InceptionResnetV2.5.tnt')
gamma = 1e-5
optimizer = torch.optim.Adam(net.parameters(), lr=gamma )
if model_path.exists():
    net = torch.load(str(model_path))
    print('load succerssful')


batch_size = 16
epoch = 3
size = 299, 299
load_flag = 0
NB = (total+batch_size-1)//batch_size
print('there are ',NB, 'batches')


# In[9]:


load_flag = 0
for e in range(epoch):
    loss_epoch = 0
    i=0
    with open('history.log', 'r') as f:
        if load_flag ==0:
            i = int(f.read())
            load_flag = 1
        
    while i < NB:

        x = torch.zeros([batch_size,3,299,299])
        y = torch.zeros([batch_size])
        for j in range(batch_size):

            try:
                img = Image.open(os.path.join(train_image_dir,np_train[i*batch_size+j]))
                img.thumbnail(size)
                img = np.array(img)
                a,c = train_transform(img)

                x[j] = transform(a)
                y[j] = tlabel[i*batch_size+j]

            except:
                print(np_train[j])
               
        inpu = x.to(device=device, dtype=dtype)  

        ltemp = y.to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        output = net.forward(inpu)
        scores = nn.functional.softmax(output,dim = 1)
        #print(output,scores,ltemp)
        loss = criterion(scores,ltemp)
        loss_epoch = loss_epoch +loss
        loss.backward()
        # Parameter update 
        optimizer.step()
        if i%100 == 0:
            print(i,loss)
        if i%1000 ==0:
            torch.save(net, 'InceptionResnetV2.5.tnt')
            with open('history.log', 'w') as f:
                f.write(str(i))
            print('save successful')
        i +=1
    print(loss_epoch/NB)


# In[ ]:


'''
see validation loss

'''
loss_val = 0
i=0

while i < NB_valid:

    x = torch.zeros([batch_size,3,299,299])
    y = torch.zeros([batch_size])

    for j in range(batch_size):
        # try...except is to prevent unexpected failure
        try:
            img = Image.open(os.path.join(train_image_dir,np_valid[(i-1)*batch_size+j]))
            # turn the image size to 299x299
            img.thumbnail(size)
            img = np.array(img)
            a,c = train_transform(img)

            x[j] = transform(a)
            y[j] = vlabel[(i-1)*batch_size+j]

        except:
            print(np_train[j])

    inpu = x.to(device=device, dtype=dtype)  

    ltemp = y.to(device=device, dtype=torch.long)
    optimizer.zero_grad()
    output = net.forward(inpu)
    #print(output)
    scores = nn.functional.softmax(output,dim = 1)
    #print(output,scores,ltemp)
    loss = criterion(scores,ltemp)
    loss_val = loss_val +loss
    i +=1
print(loss_val.data)


# In[ ]:


stop


# In[ ]:


# rename the file before save the file to prevent overwrite
torch.save(net, 'InceptionResnetV2.tnt')


# In[ ]:





# In[ ]:





# In[ ]:




