
# coding: utf-8

# # Assignment 0
# 
# Objective of this assignment is to give an overview of working with PyTorch to train, test and save your model.

# In[ ]:


from __future__ import division, print_function, unicode_literals 
import torch
import timeit
from PIL import Image
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt

# Hyper Parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 10
learning_rate = 0.001
model_name1 = 'mnist_model'
model_name2 = 'svhn_model.pkl'



# ### Loading MNIST dataset
# 
# The MNIST dataset is downloaded from internet and cached locally. The first time you run this block, it will take time to download the dataset. The dataset is saved in `datasets` directory.
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

test_dataset1 = dsets.SVHN(root='./svhn', 
                           split='test', 
                           transform=transforms.ToTensor())

test_dataset2 = MNIST_Modified(root='./mnist', 
                           train=False, 
                           transform=transforms.ToTensor())

test_loader1 = torch.utils.data.DataLoader(dataset=test_dataset1, 
                                          batch_size=batch_size, 
                                          shuffle=False)

test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset2, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# Let's see one batch of training images and test images. Notice what happens if you change the value of `batch_size` hyperparameter.

# In[24]:


def imshow(img):
    npimg = img.numpy()
    #npimg = Image.fromarray(npimg, 'RGB')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    #npimg.show()

# In[25]:


# test_dataiter = iter(test_loader)
# test_images, test_labels = test_dataiter.next()
# # print images
# print("Test images")
# imshow(torchvision.utils.make_grid(test_images))


# Neural Network Model (1 hidden layer)
class CustomResnet(nn.Module): # Extend PyTorch's Module class
    def __init__(self, num_classes = 10):
        super(CustomResnet, self).__init__() # Must call super __init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3,2,1)

         
        self.lyr1conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True) 
        self.lyr1bn1 = nn.BatchNorm2d(64)
        self.lyr1relu1 = nn.ReLU(inplace=True)
        self.lyr1conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True) 
        self.lyr1bn2 = nn.BatchNorm2d(64)

        self.lyr1relu2 = nn.ReLU(inplace=True)

        self.lyr2conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True) 
        self.lyr2bn1 = nn.BatchNorm2d(64)
        self.lyr2relu1 = nn.ReLU(inplace=True)
        self.lyr2conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True) 
        self.lyr2bn2 = nn.BatchNorm2d(64)

        self.lyr2relu2 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(4096, num_classes)  

        

        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        self.relu(out)
        out = self.maxpool(out)  

        out1 = self.lyr1conv1(out)
        out1 = self.lyr1bn1(out1)
        self.lyr1relu1(out1)
        out1 = self.lyr1conv2(out1)
        out1 = self.lyr1bn2(out1)

        out = out + out1
        self.lyr1relu2(out)

        out1 = self.lyr2conv1(out)
        out1 = self.lyr2bn1(out1)
        self.lyr2relu1(out1)
        out1 = self.lyr2conv2(out1)
        out1 = self.lyr2bn2(out1)

        out = out + out1
        self.lyr2relu2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


def test_svhn_from_mnist(model):
    correct = 0
    total = 0
    
    for images, labels in test_loader1:
        images = Variable(images)  
        labels[labels == batch_size] = 0
        labels = labels.type(torch.LongTensor)
        labels = labels.view(labels.size(0),)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()
    print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

def test_mnist_from_svhn(model):
    correct = 0
    total = 0
    
    for images, labels in test_loader2:
        images = Variable(images) 
        images = torch.cat((images, images, images), 1)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()
    print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))


start_time = timeit.default_timer()

net = CustomResnet(num_classes = 10)
net.load_state_dict(torch.load(model_name1))
net = net.eval()
print ("Using MNIST for SVHN")
test_svhn_from_mnist(net)
elapsed = timeit.default_timer() - start_time
print ('Elapsed Time: %.2f' %(elapsed)) 

net = CustomResnet(num_classes = 10)
net.load_state_dict(torch.load(model_name2))
net = net.eval()
print ("Using SVHN for MNIST")
test_mnist_from_svhn(net)
elapsed = timeit.default_timer() - start_time
print ('Elapsed Time: %.2f' %(elapsed)) 

# load the trained parameter values from the disk
# net_1.load_state_dict(torch.load('assignment0_model.pkl'))


