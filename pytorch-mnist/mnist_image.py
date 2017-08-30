# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:38:17 2017

@author: huangry
"""

import os
from PIL import Image
import struct

if __name__ == '__main__':
    def read_image_labels(filename,file_label):
        f_img = open(filename,'rb')
        index = 0
        buf_img = f_img.read()
        f_img.close()
        
        f = open(file_label,'rb')
        index1 = 0
        buf = f.read()
        f.close()
        
        magic1, images, rows, cols = struct.unpack_from('>IIII' , buf_img , index) #images is the number of images
        index += struct.calcsize('>IIII')
        
        magic2, labels = struct.unpack_from('>II' , buf , index1)
        index1 += struct.calcsize('>II')
        labelArr = [0] * labels
        
        for i in range(images):
            image = Image.new('L', (cols, rows))
            for x in range(rows):
                for y in range(cols):
                    image.putpixel((y,x),int(struct.unpack_from('>B', buf_img, index)[0]))
                    index += struct.calcsize('>B')
            
            labelArr[i] = int(struct.unpack_from('>B', buf, index1)[0]) #label
            label = str(labelArr[i])
            print(label)
            index1 += struct.calcsize('>B')
            save_dir = 'S:/pythonProgram/mnist/data/train/' + label
            if os.path.exists(save_dir) == False:
                os.mkdir(save_dir)
            
            print('save',str(i),'image')
            image.save(save_dir + '/' + str(i) + '.png')
            

    read_image_labels('S:/pythonProgram/mnist/data/train-images.idx3-ubyte','S:/pythonProgram/mnist/data/train-labels.idx1-ubyte')