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


# In[2]:


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


# In[3]:


'''
read this file to get the image file name and the label(0 or 1) and turn it into numpy
'''

label = pd.read_csv('train_ship_label.csv')

np_la = label.values
np_train = np_la[:,0]
total = len(np_train)
np_label = np_la[:,2]


# In[4]:


def normalize_MNIST_images(x):
    return 2 * x.astype(np.float64) / 255. - 1


# In[5]:


'''
set the train parameter,flag and load the model(if exists)
'''

net = InceptionResnetV2().cuda()
# we use cross entropy to calculate the loss
criterion = nn.CrossEntropyLoss()

model_path = Path('InceptionResnetV2.1.tnt')
gamma = 1e-5
optimizer = torch.optim.Adam(net.parameters(), lr=gamma )
if model_path.exists():
    net = torch.load(str(model_path))
    print('load succerssful')

batch_size = 16
epoch = 3
load_flag = 0
NB = (total+batch_size-1)//batch_size
print('there are ',NB, 'batches')
false = 0


# In[6]:


'''
when train from the beginning, the part 'with open('accurracy_0.log', 'r') as f:...' should be comment and if begin 
from the interrupt place, then cancel the comment

'''

# false_0 is the number of images which is FN and otherwise false_1
false_0 = 0
false_1 = 0
size = 299,299
for e in range(1):
    loss_epoch = 0
    i=0
    '''
    with open('accurracy_0.log', 'r') as f:
        if load_flag ==0:
            false_0 = int(f.read())
    with open('accurracy_1.log', 'r') as f:
        if load_flag ==0:
            false_1 = int(f.read())
    with open('test_his.log', 'r') as f:
        if load_flag ==0:
            i = int(f.read())
            load_flag = 1
    '''
    
    while i < NB:

        x = torch.zeros([batch_size,3,299,299])
        y = torch.zeros([batch_size])

        for j in range(batch_size):
            # try...except is to prevent unexpected failure
            try:
                img = Image.open(os.path.join(train_image_dir,np_train[i*batch_size+j]))
                # turn the image size to 299x299
                img.thumbnail(size)
                x[j] = transforms.ToTensor()(img)
                y[j] = np_label[i*batch_size+j]
            
            except:
                print(np_train[j])

        inpu = x.to(device=device, dtype=dtype)  
        ltemp = y.to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        output = net.forward(inpu)
        scores = nn.functional.softmax(output,dim = 1)
        # write the image name which is FN or FP into the file
        for n in range(batch_size):
            if scores[n,0]>scores[n,1] and ltemp[n]==1:
                with open('FP_all.log', 'a') as f:
                    f.write(np_train[(i-1)*batch_size+n]+',')
            elif scores[n,0]<scores[n,1] and ltemp[n]==0:
                with open('FN_all.log', 'a') as f:
                    f.write(np_train[(i-1)*batch_size+n]+',')
        # this part only calculate the number of FN and FP 
        '''
        if i%100 ==0:
            
            with open('accurracy_0.log', 'w') as f:
                f.write(str(false_0))
            with open('accurracy_1.log', 'w') as f:
                f.write(str(false_1))
                
            with open('test_his.log', 'w') as f:
                f.write(str(i))
            print('save successful')
        '''
            
        i +=1
    #print(loss_epoch/NB)

