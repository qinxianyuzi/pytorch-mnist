# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:53:39 2017

@author: huangry
"""

import os
import numpy as np
import gzip
import torch
from torch.autograd import Variable


if __name__ == '__main__':
    class ConvNet(torch.nn.Module):
        def __init__(self,output_dim):
            super(ConvNet,self).__init__()
            #Conv
            self.conv = torch.nn.Sequential()
            self.conv.add_module("conv_1",torch.nn.Conv2d(1,10,5)) # channels = 1,numput of map = 10,kernel_size = 5 ,conv1--10 * 24 * 24
            self.conv.add_module("maxpool_1",torch.nn.MaxPool2d(2)) # pool1--10 * 12 * 12
            self.conv.add_module("relu_1",torch.nn.ReLU()) # relu1
            self.conv.add_module("conv_2",torch.nn.Conv2d(10,20,5)) #channels = 10,numput of map = 20,kernel_size = 5 ,conv1--20 * 8 * 8
            self.conv.add_module("dropout_1",torch.nn.Dropout()) # dropout1
            self.conv.add_module("maxpool_2",torch.nn.MaxPool2d(2)) # pool2--20 * 4 * 4
            self.conv.add_module("relu_2",torch.nn.ReLU()) #relu2
            #Fc
            self.fc = torch.nn.Sequential()
            self.fc.add_module("fc1",torch.nn.Linear(20 * 4 * 4,50)) # 20 * 4 * 4 --> 50
            self.fc.add_module("relu_3",torch.nn.ReLU()) #relu3
            self.fc.add_module("dropout_2",torch.nn.Dropout()) #dropout2
            self.fc.add_module("fc2",torch.nn.Linear(50,output_dim)) #50 --> output_dim
            
        def forward(self,x):
            x = self.conv.forward(x)
            x = x.view(-1,20 * 4 * 4)
            x = self.fc.forward(x)
            return x
            
            
    def load_mnist(ntest = 10000):
        data_dir = 'S:\pythonProgram\mnist\data'
        test_images_dir = os.path.join(data_dir,'t10k-images-idx3-ubyte.gz')
        test_labels_dir = os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz')
        
        with gzip.open(test_images_dir) as fd:
            buf = fd.read()
            loaded = np.frombuffer(buf,dtype = np.uint8)
            print(len(loaded))   # output is 7840016
            teX = loaded[16:]
            teX = teX.reshape((10000,28*28)).astype(float)
            
        with gzip.open(test_labels_dir) as fd:
            buf = fd.read()
            loaded = np.frombuffer(buf,dtype = np.uint8)
            print(len(loaded))   # output is 10008
            teY = loaded[8:]
            teY  = teY.reshape((10000))
            
        teX /= 255.
        teX = teX[:ntest]
        teY = teY[:ntest]
        teY = np.asarray(teY)
        return teX, teY
        
    teX, teY = load_mnist()
    teX = teX.reshape(-1,1,28,28)
    teX = torch.from_numpy(teX).float()
    model = torch.load('S:/pythonProgram/mnist/91.pkl')
    model = model.cuda()
    correct = 0
    x = Variable(teX,requires_grad = False)
    output = model.forward(x.cuda())
    predY = output.cpu().data.numpy().argmax(axis = 1)
    correct = np.sum(predY == teY)
    total = len(teY)
    print("acc = %.2f%%" % (100. * correct / total))