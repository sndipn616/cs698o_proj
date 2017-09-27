from __future__ import division, print_function, unicode_literals
import os
import sys
import numpy as np
import torch
import cPickle
import torchvision
import torch.nn as nn
import torch.utils.data
from PIL import Image
from scipy.ndimage import imread
from collections import OrderedDict
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# All hyper parameters go in the next block

# In[ ]:
# torch.cuda.device(1)


root_dir_mnist = 'mnist'
root_dir_svhn = 'svhn'
batch_size = 10
num_epochs = 1
learning_rate = 0.01
numClasses = 10
use_gpu = False
model_file = 'custom_resnet_trained'
model_file_resnet = 'custom_resnet'
cifar_100 = 'cifar-100-python/test'

class MNIST_Modified(torchvision.datasets.MNIST):
    # def __init__(self, root, split, transform=None):

    #     self.root = root
    #     self.split = split
     def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = img.resize((32,32), Image.BILINEAR)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    
#Load Dataset
train_dataset_mnist = MNIST_Modified(root=root_dir_mnist, 
                            train=True, 
                            transform=transforms.ToTensor())

test_dataset_mnist = MNIST_Modified(root=root_dir_mnist, 
                           train=False, 
                           transform=transforms.ToTensor())

train_loader_target = torch.utils.data.DataLoader(dataset=train_dataset_mnist, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader_target = torch.utils.data.DataLoader(dataset=test_dataset_mnist, 
                                          batch_size=batch_size, 
                                          shuffle=False)

train_dataset_svhn = dsets.SVHN(root=root_dir_svhn, 
                            split='train', 
                            transform=transforms.ToTensor())

test_dataset_svhn = dsets.SVHN(root=root_dir_svhn, 
                           split='test', 
                           transform=transforms.ToTensor())

train_loader_source = torch.utils.data.DataLoader(dataset=train_dataset_svhn, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader_source = torch.utils.data.DataLoader(dataset=test_dataset_svhn, 
                                          batch_size=batch_size, 
                                          shuffle=False)


class SharedEncoder(nn.Module):
    def __init__(self, shared_encoder_out=3072):
        super(SharedEncoder, self).__init__() 

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=0,bias=True)        
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=1)

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=0,bias=True)        
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=1)

        self.fc = nn.Linear(36864, shared_encoder_out)

    def forward(self, x):

        out = self.conv1(x)
        self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        self.relu2(out)
        out = self.maxpool2(out)

        # print ("here")
        # print (out.size())
        out = out.view(out.size(0), -1)
        # print (out.size())
        out = self.fc(out)
        # print ("here2")

        return out

class PrivateEncoder(nn.Module):
    def __init__(self, target_encoder_out=3072):
        super(PrivateEncoder, self).__init__() 

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=0,bias=True)        
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=1)

        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=0,bias=True)        
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=1)

        self.fc = nn.Linear(18432, target_encoder_out)

    def forward(self, x):

        out = self.conv1(x)
        self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        self.relu2(out)
        out = self.maxpool2(out)

        # print ("here")
        # print (out.size())
        out = out.view(out.size(0), -1)
        # print (out.size())
        out = self.fc(out)
        # print ("here2")

        return out

class SharedDecoder(nn.Module):
    def __init__(self, shared_decoder_in=6144):
        super(SharedDecoder, self).__init__()

        self.fc = nn.Linear(shared_decoder_in, 300)

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=0,bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=0,bias=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.upsample = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)

        self.conv3 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=0,bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,stride=1,padding=0,bias=True)

    def forward(self, x):
        # print (x.size())
        out = self.fc(x)
        # print (out.size())
        out = out.view(out.size(0), 3,10,10)

        out = self.conv1(out)
        self.relu1(out)
        out = self.conv2(out)
        self.relu2(out)

        out = self.upsample(out)

        out = self.conv3(out)
        self.relu3(out)

        out = self.conv4(out)
        print (out.size())

        return out




def train(se,pe,sd,loader):
    # Code for training the model
    # Make sure to output a matplotlib graph of training losses
    print ("Training Resnet")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loader):  
            # Convert torch tensor to Variable           
            images = Variable(images)
            labels = Variable(labels)
            if(use_gpu):
                images=images.cuda()
                labels=labels.cuda()
            # Forward + Backward + Optimize

            optimizer.zero_grad()  # zero the gradient buffer
            outputs1 = pe(images)
            outputs2 = se(images)
            x = torch.cat((outputs1, outputs2), 1)
            outputs = sd(x)
            sys.exit()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # sys.exit()
            if (i+1) % 10 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

            if i == 500 and not use_gpu:
                break

    torch.save(model.state_dict(), model_file_resnet)


def test(model):
    # Write loops for testing the model on the test set
    # You should also print out the accuracy of the model
    correct = 0
    total = 0
    print ("Testing")
    for images, labels in test_loader:        
        
        if(use_gpu):
            images = Variable(images.cuda())
        else:
            images = Variable(images)
            
        images = torch.cat((images, images, images), 1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return (100 * correct / total)
    

pe = PrivateEncoder()
pe = pe.train()

se = SharedEncoder()
se = se.train()

sd = SharedDecoder()
sd = sd.train()

criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(pe.parameters(), lr=learning_rate)

train(se, pe, sd, train_loader_source)
# test(model)
