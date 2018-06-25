import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torch.utils.data as torch_data
import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
from model import *

import util.loader.transform as tr
from util.loader.loader import DAVIS2016
from util.metrics import runningScore
from util.loss import *
from util.utils import *
from util.helpers import *

from PIL import Image

img_rows   = 256
img_cols   = 512
batch_size = 8
lr         = 1e-4

ttransforms = transforms.Compose([tr.ScaleNRotate(), tr.RandomHorizontalFlip(), tr.Resize([img_rows, img_cols]), tr.ToTensor()])
tdataset = DAVIS2016(root='../dataset/davis.pkl', split='train', transform=ttransforms)
tdataloader = torch.utils.data.DataLoader(tdataset, batch_size=batch_size, shuffle=True, num_workers=8)

vtransforms = transforms.Compose([tr.Resize([img_rows, img_cols]), tr.ToTensor()])
vdataset = DAVIS2016(root='../dataset/davis.pkl', split='val', transform=vtransforms)
vdataloader = torch.utils.data.DataLoader(vdataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Setup Metrics
running_metrics = runningScore(pspnet_specs['n_classes'])

# Setup Model
model = pspnet()

# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model.cuda()

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
loss_fn = cross_entropy2d

best_iou = -100.0
iter = 0
writer = SummaryWriter('runs/')
for epoch in range(100):
    model.train()
    for i, data in enumerate(tdataloader):
        images = Variable(data['image'].cuda())
        labels = Variable(data['gt'].type(torch.LongTensor).cuda())
        poly_lr_scheduler(optimizer, lr, iter, lr_decay_iter=1, max_iter=1e+5)
        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(input=outputs, target=labels)
        loss.backward()
        optimizer.step()
        
        iter += 1
        writer.add_scalar('Loss', loss.data[0], iter)
        if (i+1) % 20 == 0:
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, 100, loss.data[0]))
    model.eval()
    for i_val, (data) in tqdm(enumerate(vdataloader)):
        images = Variable(data['image'].cuda(), volatile=True)
        labels = Variable(data['gt'].type(torch.LongTensor).cuda(), volatile=True)
        
        outputs = model(images)
        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels.data.cpu().numpy()
        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        print(k, v)
        if k == 'Mean IoU : \t':
            writer.add_scalar('IoU', v, epoch+1)

    running_metrics.reset()
    
    if score['Mean IoU : \t'] >= best_iou:
        best_iou = score['Mean IoU : \t'] 
        torch.save(model.state_dict(), "weight/model.pkl")
