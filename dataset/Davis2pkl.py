from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle as pkl
import cv2
from scipy.misc import imresize
from torch.utils.data import Dataset

# please write the DAVIS path
db_root_dir = ''
meanval = (104.00699, 116.66877, 122.67892)
img_size = (256, 512)
noseq = {}

X = {}
Y = {}
for fname in ['train', 'val']:
    noseq[fname] = []
    # Initialize the original DAVIS splits for training the parent network
    with open(os.path.join(db_root_dir, fname + '_seqs.txt')) as f:
        seqs = f.readlines() # list video dir name ex: train/ bear/
        img_list = []
        labels = []
        count  = []
        for seq in seqs:
            #catch all file in the dir
            images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip())))  # all photos in a dir
            images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images))
            img_list.extend(images_path)
            lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip())))
            lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
            labels.extend(lab_path)
            count.append(len(images_path))
        end = count[:]
        for i in range(1,len(end)):
            end[i]+=end[i-1]
        for i in end:
            noseq[fname].extend([i-1,i-2])
        
        X[fname] = np.zeros([end[-1], img_size[0], img_size[1], 3], np.uint8)
        Y[fname] = np.zeros([end[-1], img_size[0], img_size[1]], np.float32)

        # read file
        for idx in range(len(img_list)):
            img = cv2.imread(os.path.join(db_root_dir, img_list[idx]))
            print(img_list[idx])
            if labels[idx] is not None:
                lbl = cv2.imread(os.path.join(db_root_dir, labels[idx]), 0)
            #if no gt then do 0
            else:
                gt = np.zeros(img.shape[:-1], dtype=np.uint8)
     
            img = cv2.resize(img, (img_size[1], img_size[0]))
            if labels[idx] is not None:
                lbl = cv2.resize(lbl, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)

            ''' 
            img = np.array(img, dtype=np.float32)
            img = np.subtract(img, np.array(meanval, dtype=np.float32))
            '''
            
            if labels[idx] is not None:
                gt = np.array(lbl, dtype=np.float32)
                gt = gt/np.max([gt.max(), 1e-8])

            X[fname][idx] = img
            Y[fname][idx] = gt
               
            
with open('davis.pkl', 'wb') as f:
    pkl.dump({ 'train': {'images': X['train'], 'labels': Y['train'], 'noseq': noseq['train']},
               'val'  : {'images': X['val']  , 'labels': Y['val']  , 'noseq': noseq['val']},
             }, f, pkl.HIGHEST_PROTOCOL)

