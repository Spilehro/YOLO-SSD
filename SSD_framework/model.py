import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    1
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    batch_size = ann_confidence.shape[0]
    num_of_classes = ann_confidence.shape[2]
    cell_number = ann_confidence.shape[1]
    
    #TODO: write a loss function for SSD
    #
    ann_confidence = ann_confidence.reshape(batch_size*cell_number,num_of_classes)
    ann_box = ann_box.reshape(batch_size*cell_number, 4)
    pred_confidence = pred_confidence.reshape(batch_size*cell_number,num_of_classes)
    pred_box = pred_box.reshape(batch_size*cell_number, 4)
    
    length = ann_confidence.shape[0]
    object_indices = torch.ones(length,dtype=torch.int32)
    background = torch.zeros(length,num_of_classes,dtype=torch.float32).cuda()
    background[:,-1]=1
    object_indices [torch.sum(background.eq(ann_confidence),dim=1)==num_of_classes]=0
    
    
    # for i in range(length):
    #     if not (torch.all(background.eq(ann_confidence[i]))):
    #         indices[i]=1
            
    gt_object_box = ann_box[object_indices==1]
    gt_object_confidence = ann_confidence[object_indices==1]
    pred_object_box = pred_box[object_indices==1]
    pred_object_confidence = pred_confidence[object_indices==1]
    # gt_background_box = ann_box[indices==0]
    gt_background_confidence = ann_confidence[object_indices==0]
    # pred_background_box = pred_box[indices==0]
    pred_background_confidence = pred_confidence[object_indices==0]
    l_cls = F.binary_cross_entropy(pred_object_confidence,gt_object_confidence)+ 3*F.binary_cross_entropy(pred_background_confidence,gt_background_confidence)
    l_box = F.smooth_l1_loss(pred_object_box,gt_object_box)
    loss = l_cls+l_box
    return loss

    #For confidence (class labels), use cross entropy (F.binary_cross_entropy)
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.

def Layer (inc,outc,ks,s,p):
    return nn.Sequential(
        nn.Conv2d(inc,outc,ks,s,p),
        nn.BatchNorm2d(outc),
        nn.ReLU(inplace=True)
    )

def conv_branch (inc=256,s=1,p=0):
    return nn.Sequential(
        nn.Conv2d(inc,inc,1,1,0),
        nn.Conv2d(inc,inc,3,s,p)
    )

class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        self.layers = nn.ModuleList()

        #TODO: define layers
        self.layers.append( Layer(3,64,3,2,1))
        self.layers.append( Layer(64,64,3,1,1))
        self.layers.append( Layer(64,64,3,1,1))
        self.layers.append( Layer(64,128,3,2,1))
        self.layers.append( Layer(128,128,3,1,1))
        self.layers.append( Layer(128,128,3,1,1))
        self.layers.append( Layer(128,256,3,2,1))
        self.layers.append( Layer(256,256,3,1,1))
        self.layers.append( Layer(256,256,3,1,1))
        self.layers.append( Layer(256,512,3,2,1))
        self.layers.append( Layer(512,512,3,1,1))
        self.layers.append( Layer(512,512,3,1,1))
        self.layers.append( Layer(512,256,3,2,1))

        self.conv_branch1 = conv_branch(s=2,p=1)
        self.conv_branch2 = conv_branch()
        self.conv_branch3 = conv_branch()

        self.mainbb = nn.Conv2d(256,16,kernel_size=1,stride=1,padding=0)
        self.mainconf = nn.Conv2d(256,16,kernel_size=1,stride=1,padding=0)

        self.red1 = nn.Conv2d(256,16,kernel_size=3,stride=1,padding=1)
        self.red2 = nn.Conv2d(256,16,kernel_size=3,stride=1,padding=1)
        self.red3 = nn.Conv2d(256,16,kernel_size=3,stride=1,padding=1)

        self.blue1 = nn.Conv2d(256,16,kernel_size=3,stride=1,padding=1)
        self.blue2 = nn.Conv2d(256,16,kernel_size=3,stride=1,padding=1)
        self.blue3 = nn.Conv2d(256,16,kernel_size=3,stride=1,padding=1)

        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        for i in range(len(self.layers)):
            x=self.layers[i](x)
         
        x_main = self.conv_branch1(x)
        x_main2 = self.conv_branch2(x_main)
        x_main3 = self.conv_branch3(x_main2)
        
        bb_main = self.mainbb(x_main3)
        bb_main = bb_main.reshape((bb_main.shape[0],bb_main.shape[1],bb_main.shape[2]*bb_main.shape[3]))
        conf_main = self.mainconf(x_main3)
        conf_main = conf_main.reshape((conf_main.shape[0],conf_main.shape[1],conf_main.shape[2]*conf_main.shape[3]))
        
        bb1 = self.red1(x)
        bb1 = bb1.reshape((bb1.shape[0],bb1.shape[1],bb1.shape[2]*bb1.shape[3]))
        conf1 = self.blue1(x)
        conf1 = conf1.reshape((conf1.shape[0],conf1.shape[1],conf1.shape[2]*conf1.shape[3]))

        bb2 = self.red2(x_main)
        bb2 = bb2.reshape((bb2.shape[0],bb2.shape[1],bb2.shape[2]*bb2.shape[3]))
        conf2 = self.blue2(x_main)
        conf2 = conf2.reshape((conf2.shape[0],conf2.shape[1],conf2.shape[2]*conf2.shape[3]))

        bb3 = self.red3(x_main2)
        bb3 = bb3.reshape((bb3.shape[0],bb3.shape[1],bb3.shape[2]*bb3.shape[3]))
        conf3 = self.blue3(x_main2)
        conf3 = conf3.reshape((conf3.shape[0],conf3.shape[1],conf3.shape[2]*conf3.shape[3]))



        #remember to apply softmax to confidence! Which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        bboxes = torch.cat([bb1,bb2,bb3,bb_main],dim=2)
        bboxes = bboxes.permute(0,2,1)

        confidence = torch.cat([conf1,conf2,conf3,conf_main],dim=2)
        confidence = confidence.permute(0,2,1)

        total_cells = bboxes.shape[1]*4
        defult_bb_num = 4
        bboxes = bboxes.reshape((bboxes.shape[0],total_cells,defult_bb_num))
        confidence = confidence.reshape((confidence.shape[0],total_cells,defult_bb_num))
        confidence = torch.softmax(confidence,dim=2)


        return confidence,bboxes










