# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:53:03 2017

@author: huangry
"""

import os
import gzip
import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim

import random
import cv2

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
        
            
    def load_mnist(ntrain = 60000, ntest = 10000):
        data_dir = 'S:\pythonProgram\mnist\data'
        train_images_dir = os.path.join(data_dir,'train-images-idx3-ubyte.gz')
        train_labels_dir = os.path.join(data_dir,'train-labels-idx1-ubyte.gz')
        test_images_dir = os.path.join(data_dir,'t10k-images-idx3-ubyte.gz')
        test_labels_dir = os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz')
        
        with gzip.open(train_images_dir) as fd:
            buf = fd.read()
            loaded = np.frombuffer(buf,dtype = np.uint8)
            print(len(loaded))   # output is 47040016
            trX = loaded[16:]
            trX = trX.reshape((60000,28*28)).astype(float)
            
        with gzip.open(train_labels_dir) as fd:
            buf = fd.read()
            loaded = np.frombuffer(buf,dtype = np.uint8)
            print(len(loaded))   # output is 60008
            trY = loaded[8:]
            trY  = trY.reshape((60000))
            
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
            
        trX /= 255.
        teX /= 255.
        trX = trX[:ntrain]
        trY = trY[:ntrain]
        teX = teX[:ntest]
        teY = teY[:ntest]
        trY = np.asarray(trY)
        teY = np.asarray(teY)
        return trX, teX, trY, teY
        
    def load_mnist_images(images_dir):
        img_all_path = []
        base_dir = images_dir
        dirs = os.listdir(base_dir) 
        for dir in dirs:
            img_path = base_dir + '/' + dir
            img_paths = os.listdir(img_path)
            for img_dirs in img_paths:
                img_dir = img_path + '/' + img_dirs
                img_label = []
                img_label.append(img_dir) #img dir
                img_label.append(dir) #img label
                img_all_path.append(img_label)
        random.shuffle(img_all_path)

        data_imgs = []
        data_label = []
        for item in img_all_path:
            img_dir = item[0]
            label = item[1]
            img = cv2.imread(img_dir,0) / 255.0
            img = cv2.resize(img,(28,28))
            data_imgs.append(img[None,...])
            data_label.append(int(label))
            
        data_imgs = np.concatenate(data_imgs,axis = 0)
        data_label = np.array(data_label)
        return data_imgs, data_label
        


    
    torch.manual_seed(42)
#    trX, teX, trY, teY = load_mnist()
    trX, trY = load_mnist_images('S:/pythonProgram/mnist/data/train')
    teX, teY = load_mnist_images('S:/pythonProgram/mnist/data/test')
    trX = trX.reshape(-1,1,28,28) # -1保留该通道原有大小
    teX = teX.reshape(-1,1,28,28)
    
    trX = torch.from_numpy(trX).float() # numpy to tensor
    teX = torch.from_numpy(teX).float()
    trY = torch.from_numpy(trY).long()
    
    n_examples = len(trX) # the num of examples is 60000
    n_classes = 10
    model = ConvNet(n_classes)
    model = model.cuda()
    loss = torch.nn.CrossEntropyLoss(size_average = True)
    optimizer = optim.SGD(model.parameters(), lr = 0.01,momentum = 0.9)
    batch_size = 100
    epochs = 500
    for i in range(epochs):
        cost = 0.
        num_batches = n_examples / batch_size
        num_batches = int(num_batches) # 600
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            x_val = trX[start:end]
            y_val = trY[start:end]
            x = Variable(x_val, requires_grad = False)
            y = Variable(y_val, requires_grad = False)
            
            optimizer.zero_grad()
            fx = model.forward(x.cuda())
            output = loss.forward(fx,y.cuda())
            output.backward()
            optimizer.step()
            cost += output.data[0]

        if (i - 1) % 10 == 0:
            torch.save(model,'S:\pythonProgram\mnist\%d.pkl' % (i)) # save net and model params
        correct = 0
        x = Variable(teX,requires_grad = False)
        output = model.forward(x.cuda())
        predY = output.cpu().data.numpy().argmax(axis = 1)
        correct = np.sum(predY == teY)
        total = len(teY)
        print("Epoch %d, cost = %f, acc = %.4f%%" % (i + 1,cost / num_batches, 100. * correct / total))