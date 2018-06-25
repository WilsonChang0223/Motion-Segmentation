import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.autograd import Variable

pspnet_specs = {
    'n_classes': 2,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}

class ConvGRUCell(nn.Module):
    
    def __init__(self,input_size,hidden_size,kernel_size):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=0.5)
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,2 * self.hidden_size,self.kernel_size,padding=self.kernel_size//2)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,self.hidden_size,self.kernel_size,padding=self.kernel_size//2) 
        dtype            = torch.FloatTensor
    
    def forward(self,input,hidden):
        
        if hidden is None:
           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
           hidden    = Variable(torch.zeros(size_h)).cuda()
        c1           = self.ConvGates(torch.cat((input,hidden),1))
        (rt,ut)      = c1.chunk(2, 1)
        reset_gate   = self.dropout(f.sigmoid(rt))
        update_gate  = self.dropout(f.sigmoid(ut))
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        ct           = f.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        return next_h

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                            padding=padding, stride=stride, bias=bias, dilation=dilation)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes):
        super(pyramidPooling, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]

        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            out = F.avg_pool2d(x, int(h/pool_size), int(h/pool_size), 0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear')
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.n_classes = pspnet_specs['n_classes']
        
        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        # Pyramid Pooling Module
        self.pyramid_pooling = pyramidPooling(2048, [6, 3, 2, 1])
        self.final = conv2DBatchNormRelu(4096, 256, 3, 1, 1, False)

    def forward(self, x):
        inp_shape = x.shape[2:]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pyramid_pooling(x)

        return self.final(x), inp_shape

class MotionNet(nn.Module):
    def __init__(self):
        super(MotionNet, self).__init__()
        self.n_classes = pspnet_specs['n_classes']
        
        self.preprocess = nn.Sequential(nn.Conv2d(9, 32, 8, padding=3, stride=2),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 64, 6, padding=2, stride=2),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 128, 4, padding=1, stride=2),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                       )
                                       
    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.preprocess(x)
        return x

class ClassNet(nn.Module):
    def __init__(self):
        super(ClassNet, self).__init__()
        self.n_classes = pspnet_specs['n_classes'] 
                                       
        self.main = nn.Sequential(nn.Conv2d(128+256, 128, 5, padding=2),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 64, 3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                 )
       
        # Final conv layers
        # self.cbr_final = conv2DBatchNormRelu(512, 256, 3, 1, 1, False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        self.classification = nn.Conv2d(64, self.n_classes, 1, 1, 0)

    def forward(self, motionx, segx, inp_shape):
        x = self.main(torch.cat((motionx, segx), dim=1))
        # y = self.cbr_final(y)
        x = self.dropout(x)

        x = self.classification(x)
        x = F.upsample(x, size=inp_shape, mode='bilinear')
        return x
