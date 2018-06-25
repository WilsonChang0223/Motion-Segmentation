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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
vdataloader = torch.utils.data.DataLoader(vdataset, batch_size=1, shuffle=False, num_workers=8)

# Setup Metrics
running_metrics = runningScore(pspnet_specs['n_classes'])

# setup Model
base_net = BaseNet()
motion_net = MotionNet()
class_net = ClassNet()

base_net.cuda()
motion_net.cuda()
class_net.cuda()

base_opt = torch.optim.Adam(base_net.parameters(), lr=lr,     betas=(0.5, 0.999))
motion_opt = torch.optim.Adam(motion_net.parameters(), lr=5* lr, betas=(0.5, 0.999))
class_opt = torch.optim.Adam(class_net.parameters(), lr=lr,   betas=(0.5, 0.999))

loss_fn = cross_entropy2d

iter = 0
writer = SummaryWriter('runs/')
best_iou = -100.0
for epoch in range(100):
    base_net.train()
    motion_net.train()
    class_net.train()
    for i, (data1, data2, data3) in enumerate(tdataloader):
        images1 = Variable(data1['image'].cuda())
        images2 = Variable(data2['image'].cuda())
        images3 = Variable(data3['image'].cuda())
        labels  = Variable(data3['gt'].type(torch.LongTensor).cuda())
        poly_lr_scheduler(base_opt , lr, iter, lr_decay_iter=1, max_iter=1e+5)
        poly_lr_scheduler(class_opt, lr, iter, lr_decay_iter=1, max_iter=1e+5)
        poly_lr_scheduler(motion_opt, 5* lr, iter, lr_decay_iter=1, max_iter=1e+5)
        iter += 1

        seg, inp = base_net(images3)
        motion = motion_net(images1, images2, images3)
        outputs = class_net(motion, seg, inp)
        
        base_opt.zero_grad()
        motion_opt.zero_grad()
        class_opt.zero_grad()

        loss = loss_fn(input=outputs, target=labels)
        loss.backward()
        class_opt.step()
        base_opt.step()
        motion_opt.step()
 
        writer.add_scalar('Loss', loss.data[0], iter)
        if (i+1) % 20 == 0:
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, 100, loss.data[0]))
    base_net.eval()
    motion_net.eval()
    class_net.eval()
    for i_val, (data1, data2, data3) in tqdm(enumerate(vdataloader)):
        images1 = Variable(data1['image'].cuda())
        images2 = Variable(data2['image'].cuda())
        images3 = Variable(data3['image'].cuda())
        labels  = Variable(data3['gt'].type(torch.LongTensor).cuda())
        
        seg, inp = base_net(images3)
        motion = motion_net(images1, images2, images3)
        outputs = class_net(motion, seg, inp)
        
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
        torch.save(base_net.state_dict(), "weight/base_net.pkl")
        torch.save(class_net.state_dict(), "weight/class_net.pkl")
        torch.save(motion_net.state_dict(), "weight/motion_net.pkl")
