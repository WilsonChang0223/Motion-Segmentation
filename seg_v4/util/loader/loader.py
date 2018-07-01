import pickle 
import os
import numpy as np
import cv2
from scipy.misc import imresize
from util.helpers import *
from torch.utils.data import Dataset

class DAVIS2016(Dataset):
    def __init__(self, root, split='train', img_size=(256, 512), transform=None):
        self.transform = transform
        self.img_size = img_size
        self.meanval = (104.00699, 116.66877, 122.67892)

        with open(root, 'rb') as fp:
            contents = pickle.load(fp)
            tmp = contents[split]
            self.data = tmp['images']
            self.label= tmp['labels']
            self.noseq= tmp['noseq']

        self.shape = self.data.shape

    def __getitem__(self, index):
        '''
        img, lbl= self.data[index], self.label[index]
        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))
        '''
        if index not in self.noseq:
            img1, lbl1= self.data[index]  , self.label[index]
            img2, lbl2= self.data[index+1], self.label[index+1]
            img3, lbl3= self.data[index+2], self.label[index+2]
        else:
            img1, lbl1= self.data[index-2], self.label[index-2]
            img2, lbl2= self.data[index-1], self.label[index-1]
            img3, lbl3= self.data[index]  , self.label[index]

        img1 = np.array(img1, dtype=np.float32)
        img1 = np.subtract(img1, np.array(self.meanval, dtype=np.float32))
        img2 = np.array(img2, dtype=np.float32)
        img2 = np.subtract(img2, np.array(self.meanval, dtype=np.float32))
        img3 = np.array(img3, dtype=np.float32)
        img3 = np.subtract(img3, np.array(self.meanval, dtype=np.float32))
        
        lbl1 = np.expand_dims(lbl1, axis=2)
        lbl2 = np.expand_dims(lbl2, axis=2)
        lbl3 = np.expand_dims(lbl3, axis=2)
        
        img  = np.concatenate((img1,img2,img3),2)
        lbl  = np.concatenate((lbl1,lbl2,lbl3),2)
        
        sample = {'image': img, 'gt': lbl}
        if self.transform is not None:
            sample = self.transform(sample)
       
        sample1 = {'image': sample['image'][0:3, :, :], 'gt': sample['gt'][0, :, :]}
        sample2 = {'image': sample['image'][3:6, :, :], 'gt': sample['gt'][1, :, :]}
        sample3 = {'image': sample['image'][6:9, :, :], 'gt': sample['gt'][2, :, :]}

        return sample1, sample2, sample3
    
    def __len__(self):
        return self.data.shape[0]

