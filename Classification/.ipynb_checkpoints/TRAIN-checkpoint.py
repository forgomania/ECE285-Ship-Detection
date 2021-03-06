{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "from skimage.data import imread\n",
    "import matplotlib.pyplot as plt\n",
    "import torch.nn as nn \n",
    "import torch.nn.functional as F\n",
    "from torchvision import transforms\n",
    "\n",
    "from PIL import Image\n",
    "import os\n",
    "import torch\n",
    "from inceptionresnetv2.pytorch_load import InceptionResnetV2\n",
    "from pathlib import Path\n",
    "import random\n",
    "import cv2\n",
    "from torchvision.transforms import ToTensor, Normalize, Compose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "    Implementation from  https://github.com/ternaus/robot-surgery-segmentation\n",
    "    Data augemntation methods\n",
    "    \n",
    "\"\"\"\n",
    "\n",
    "class DualCompose:\n",
    "    def __init__(self, transforms):\n",
    "        self.transforms = transforms\n",
    "\n",
    "    def __call__(self, x, mask=None):\n",
    "        for t in self.transforms:\n",
    "            x, mask = t(x, mask)\n",
    "        return x, mask\n",
    "    \n",
    "class OneOf:\n",
    "    def __init__(self, transforms, prob=0.5):\n",
    "        self.transforms = transforms\n",
    "        self.prob = prob\n",
    "\n",
    "    def __call__(self, x, mask=None):\n",
    "        if random.random() < self.prob:\n",
    "            t = random.choice(self.transforms)\n",
    "            t.prob = 1.\n",
    "            x, mask = t(x, mask)\n",
    "        return x, mask\n",
    "\n",
    "class OneOrOther:\n",
    "    def __init__(self, first, second, prob=0.5):\n",
    "        self.first = first\n",
    "        first.prob = 1.\n",
    "        self.second = second\n",
    "        second.prob = 1.\n",
    "        self.prob = prob\n",
    "\n",
    "    def __call__(self, x, mask=None):\n",
    "        if random.random() < self.prob:\n",
    "            x, mask = self.first(x, mask)\n",
    "        else:\n",
    "            x, mask = self.second(x, mask)\n",
    "        return x, mask\n",
    "\n",
    "\n",
    "class ImageOnly:\n",
    "    def __init__(self, trans):\n",
    "        self.trans = trans\n",
    "\n",
    "    def __call__(self, x, mask=None):\n",
    "        return self.trans(x), mask\n",
    "\n",
    "\n",
    "class VerticalFlip:\n",
    "    def __init__(self, prob=0.5):\n",
    "        self.prob = prob\n",
    "\n",
    "    def __call__(self, img, mask=None):\n",
    "        if random.random() < self.prob:\n",
    "            img = cv2.flip(img, 0)\n",
    "            if mask is not None:\n",
    "                mask = cv2.flip(mask, 0)\n",
    "        return img, mask\n",
    "\n",
    "\n",
    "class HorizontalFlip:\n",
    "    def __init__(self, prob=0.5):\n",
    "        self.prob = prob\n",
    "\n",
    "    def __call__(self, img, mask=None):\n",
    "        if random.random() < self.prob:\n",
    "            img = cv2.flip(img, 1)\n",
    "            if mask is not None:\n",
    "                mask = cv2.flip(mask, 1)\n",
    "        return img, mask\n",
    "\n",
    "\n",
    "class RandomFlip:\n",
    "    def __init__(self, prob=0.5):\n",
    "        self.prob = prob\n",
    "\n",
    "    def __call__(self, img, mask=None):\n",
    "        if random.random() < self.prob:\n",
    "            d = random.randint(-1, 1)\n",
    "            img = cv2.flip(img, d)\n",
    "            if mask is not None:\n",
    "                mask = cv2.flip(mask, d)\n",
    "        return img, mask\n",
    "\n",
    "\n",
    "class Transpose:\n",
    "    def __init__(self, prob=0.5):\n",
    "        self.prob = prob\n",
    "\n",
    "    def __call__(self, img, mask=None):\n",
    "        if random.random() < self.prob:\n",
    "            img = img.transpose(1, 0, 2)\n",
    "            if mask is not None:\n",
    "                mask = mask.transpose(1, 0, 2)\n",
    "        return img, mask\n",
    "\n",
    "\n",
    "class RandomRotate90:\n",
    "    def __init__(self, prob=0.5):\n",
    "        self.prob = prob\n",
    "\n",
    "    def __call__(self, img, mask=None):\n",
    "        if random.random() < self.prob:\n",
    "            factor = random.randint(0, 4)\n",
    "            img = np.rot90(img, factor)\n",
    "            if mask is not None:\n",
    "                mask = np.rot90(mask, factor)\n",
    "        return img.copy(), mask.copy()\n",
    "\n",
    "\n",
    "class Rotate:\n",
    "    def __init__(self, limit=90, prob=0.5):\n",
    "        self.prob = prob\n",
    "        self.limit = limit\n",
    "\n",
    "    def __call__(self, img, mask=None):\n",
    "        if random.random() < self.prob:\n",
    "            angle = random.uniform(-self.limit, self.limit)\n",
    "\n",
    "            height, width = img.shape[0:2]\n",
    "            mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)\n",
    "            img = cv2.warpAffine(img, mat, (height, width),\n",
    "                                 flags=cv2.INTER_LINEAR,\n",
    "                                 borderMode=cv2.BORDER_REFLECT_101)\n",
    "            if mask is not None:\n",
    "                mask = cv2.warpAffine(mask, mat, (height, width),\n",
    "                                      flags=cv2.INTER_LINEAR,\n",
    "                                      borderMode=cv2.BORDER_REFLECT_101)\n",
    "\n",
    "        return img, mask\n",
    "\n",
    "\n",
    "class RandomCrop:\n",
    "    def __init__(self, size):\n",
    "        self.h = size[0]\n",
    "        self.w = size[1]\n",
    "\n",
    "    def __call__(self, img, mask=None):\n",
    "        height, width, _ = img.shape\n",
    "\n",
    "        h_start = np.random.randint(0, height - self.h)\n",
    "        w_start = np.random.randint(0, width - self.w)\n",
    "\n",
    "        img = img[h_start: h_start + self.h, w_start: w_start + self.w,:]\n",
    "\n",
    "        assert img.shape[0] == self.h\n",
    "        assert img.shape[1] == self.w\n",
    "\n",
    "        if mask is not None:\n",
    "            if mask.ndim == 2:\n",
    "                mask = np.expand_dims(mask, axis=2)\n",
    "            mask = mask[h_start: h_start + self.h, w_start: w_start + self.w,:]\n",
    "\n",
    "        return img, mask\n",
    "\n",
    "class RandomBrightness:\n",
    "    def __init__(self, limit=0.1, prob=0.5):\n",
    "        self.limit = limit\n",
    "        self.prob = prob\n",
    "\n",
    "    def __call__(self, img):\n",
    "        if random.random() < self.prob:\n",
    "            alpha = 1.0 + self.limit * random.uniform(-1, 1)\n",
    "\n",
    "            maxval = np.max(img[..., :3])\n",
    "            dtype = img.dtype\n",
    "            img[..., :3] = clip(alpha * img[..., :3], dtype, maxval)\n",
    "        return img\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "using device: cuda\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "set the environment and data file path\n",
    "'''\n",
    "\n",
    "USE_GPU = True\n",
    "\n",
    "dtype = torch.float32 # we will be using float for most data\n",
    "# the default is gpu but if gpu is not available,then use cpu\n",
    "if USE_GPU and torch.cuda.is_available():\n",
    "    device = torch.device('cuda')\n",
    "else:\n",
    "    device = torch.device('cpu')\n",
    "\n",
    "# show which device to use\n",
    "print('using device:', device)\n",
    "\n",
    "ship_dir = '/datasets/ee285f-public/airbus_ship_detection/'\n",
    "train_image_dir = os.path.join(ship_dir, 'train_v2')\n",
    "test_image_dir = os.path.join(ship_dir, 'test_v2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "180000\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "read this file to get the image file name and the label(0 or 1) and turn it into numpy\n",
    "then seperate it into train and validation\n",
    "'''\n",
    "\n",
    "label = pd.read_csv('train_ship_label.csv')\n",
    "\n",
    "np_la = label.values\n",
    "np_train = np_la[0:180000,0]\n",
    "np_valid= np_la[180000:192556,0]\n",
    "total = len(np_train)\n",
    "tlabel = np_la[0:180000,2]\n",
    "vlabel = np_la[180000:192556,2]\n",
    "print(total)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "train_transform is to do some augmentation the the image\n",
    "transform is to convert the data from (0 ,255) to (0,1)\n",
    "'''\n",
    "\n",
    "train_transform = DualCompose([\n",
    "        HorizontalFlip(),\n",
    "        VerticalFlip(),\n",
    "        #ImageOnly(RandomBrightness()),\n",
    "        #ImageOnly(RandomContrast()),\n",
    "])\n",
    "transform = Compose([\n",
    "        ToTensor(),\n",
    "        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "        ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "load succerssful\n",
      "11250\n"
     ]
    }
   ],
   "source": [
    "'''\n",
    "set the train parameter,flag and load the model(if exist)\n",
    "'''\n",
    "\n",
    "\n",
    "net = InceptionResnetV2().cuda()\n",
    "# we use cross entropy to calculate the loss\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "\n",
    "model_path = Path('InceptionResnetV2.5.tnt')\n",
    "gamma = 1e-5\n",
    "optimizer = torch.optim.Adam(net.parameters(), lr=gamma )\n",
    "if model_path.exists():\n",
    "    net = torch.load(str(model_path))\n",
    "    print('load succerssful')\n",
    "\n",
    "\n",
    "batch_size = 16\n",
    "epoch = 3\n",
    "size = 299, 299\n",
    "load_flag = 0\n",
    "NB = (total+batch_size-1)//batch_size\n",
    "print('there are ',NB, 'batches')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 tensor(0.7054, device='cuda:0')\n",
      "save successful\n",
      "100 tensor(0.7075, device='cuda:0')\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0mTraceback (most recent call last)",
      "\u001b[0;32m<ipython-input-9-9f557e67ba52>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     16\u001b[0m             \u001b[0;31m#try:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 17\u001b[0;31m             \u001b[0mimg\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mImage\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mos\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mjoin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain_image_dir\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mnp_train\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0mbatch_size\u001b[0m\u001b[0;34m+\u001b[0m\u001b[0mj\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     18\u001b[0m             \u001b[0;31m#img = Image.open(os.path.join(train_image_dir,np_train[j]))\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     19\u001b[0m             \u001b[0mimg\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mthumbnail\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msize\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/opt/conda/lib/python3.6/site-packages/PIL/Image.py\u001b[0m in \u001b[0;36mopen\u001b[0;34m(fp, mode)\u001b[0m\n\u001b[1;32m   2546\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2547\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mfilename\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2548\u001b[0;31m         \u001b[0mfp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbuiltins\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"rb\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2549\u001b[0m         \u001b[0mexclusive_fp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2550\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "load_flag = 0\n",
    "for e in range(epoch):\n",
    "    loss_epoch = 0\n",
    "    i=0\n",
    "    with open('history.log', 'r') as f:\n",
    "        if load_flag ==0:\n",
    "            i = int(f.read())\n",
    "            load_flag = 1\n",
    "        \n",
    "    while i < NB:\n",
    "\n",
    "        x = torch.zeros([batch_size,3,299,299])\n",
    "        y = torch.zeros([batch_size])\n",
    "        for j in range(batch_size):\n",
    "\n",
    "            try:\n",
    "                img = Image.open(os.path.join(train_image_dir,np_train[i*batch_size+j]))\n",
    "                img.thumbnail(size)\n",
    "                img = np.array(img)\n",
    "                a,c = train_transform(img)\n",
    "\n",
    "                x[j] = transform(a)\n",
    "                y[j] = tlabel[i*batch_size+j]\n",
    "\n",
    "            except:\n",
    "                print(np_train[j])\n",
    "               \n",
    "        inpu = x.to(device=device, dtype=dtype)  \n",
    "\n",
    "        ltemp = y.to(device=device, dtype=torch.long)\n",
    "        optimizer.zero_grad()\n",
    "        output = net.forward(inpu)\n",
    "        scores = nn.functional.softmax(output,dim = 1)\n",
    "        #print(output,scores,ltemp)\n",
    "        loss = criterion(scores,ltemp)\n",
    "        loss_epoch = loss_epoch +loss\n",
    "        loss.backward()\n",
    "        # Parameter update \n",
    "        optimizer.step()\n",
    "        if i%100 == 0:\n",
    "            print(i,loss)\n",
    "        if i%1000 ==0:\n",
    "            torch.save(net, 'InceptionResnetV2.5.tnt')\n",
    "            with open('history.log', 'w') as f:\n",
    "                f.write(str(i))\n",
    "            print('save successful')\n",
    "        i +=1\n",
    "    print(loss_epoch/NB)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "see validation loss\n",
    "\n",
    "'''\n",
    "loss_val = 0\n",
    "i=0\n",
    "\n",
    "while i < NB_valid:\n",
    "\n",
    "    x = torch.zeros([batch_size,3,299,299])\n",
    "    y = torch.zeros([batch_size])\n",
    "\n",
    "    for j in range(batch_size):\n",
    "        # try...except is to prevent unexpected failure\n",
    "        try:\n",
    "            img = Image.open(os.path.join(train_image_dir,np_valid[(i-1)*batch_size+j]))\n",
    "            # turn the image size to 299x299\n",
    "            img.thumbnail(size)\n",
    "            img = np.array(img)\n",
    "            a,c = train_transform(img)\n",
    "\n",
    "            x[j] = transform(a)\n",
    "            y[j] = vlabel[(i-1)*batch_size+j]\n",
    "\n",
    "        except:\n",
    "            print(np_train[j])\n",
    "\n",
    "    inpu = x.to(device=device, dtype=dtype)  \n",
    "\n",
    "    ltemp = y.to(device=device, dtype=torch.long)\n",
    "    optimizer.zero_grad()\n",
    "    output = net.forward(inpu)\n",
    "    #print(output)\n",
    "    scores = nn.functional.softmax(output,dim = 1)\n",
    "    #print(output,scores,ltemp)\n",
    "    loss = criterion(scores,ltemp)\n",
    "    loss_val = loss_val +loss\n",
    "    i +=1\n",
    "print(loss_val.data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "stop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# rename the file before save the file to prevent overwrite\n",
    "torch.save(net, 'InceptionResnetV2.tnt')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
