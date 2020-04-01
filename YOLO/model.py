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




def YOLO_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from YOLO, [batch_size, 5,5, num_of_classes]
    #pred_box        -- the predicted bounding boxes from YOLO, [batch_size, 5,5, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, 5,5, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, 5,5, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for YOLO
    #
    #For confidence (class labels), use cross entropy (F.binary_cross_entropy)
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    batch_size = ann_confidence.shape[0]
    num_of_classes = ann_confidence.shape[3]
    cell_number = ann_confidence.shape[1]
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*5*5, num_of_classes]
    #and reshape box to [batch_size*5*5, 4].

    # ann_confidence = torch.reshape(ann_confidence,(batch_size*cell_number*cell_number,num_of_classes))
    # ann_box = torch.reshape(ann_box,(batch_size*cell_number*cell_number, 4))
    # pred_confidence = torch.reshape(pred_confidence,(batch_size*cell_number*cell_number,num_of_classes))
    # pred_box = torch.reshape(pred_box,(batch_size*cell_number*cell_number, 4))
    
    ann_confidence = ann_confidence.reshape(batch_size*cell_number*cell_number,num_of_classes)
    ann_box = ann_box.reshape(batch_size*cell_number*cell_number, 4)
    pred_confidence = pred_confidence.reshape(batch_size*cell_number*cell_number,num_of_classes)
    pred_box = pred_box.reshape(batch_size*cell_number*cell_number, 4)
    
    length = ann_confidence.shape[0]
    indices=torch.zeros(length,dtype=torch.int32)
    background = torch.tensor([0.,0.,0.,1.],dtype=torch.float32).cuda()
    
    for i in range(length):
        if not (torch.all(background.eq(ann_confidence[i]))):
            indices[i]=1
            
    gt_object_box = ann_box[indices==1]
    gt_object_confidence = ann_confidence[indices==1]
    pred_object_box = pred_box[indices==1]
    pred_object_confidence = pred_confidence[indices==1]
    # gt_background_box = ann_box[indices==0]
    gt_background_confidence = ann_confidence[indices==0]
    # pred_background_box = pred_box[indices==0]
    pred_background_confidence = pred_confidence[indices==0]
    l_cls = F.binary_cross_entropy(pred_object_confidence,gt_object_confidence)+ 3*F.binary_cross_entropy(pred_background_confidence,gt_background_confidence)
    l_box = F.smooth_l1_loss(pred_object_box,gt_object_box)
    loss = l_cls+l_box

    return  loss

    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.


def Layer (inc,outc,ks,s,p):
    return nn.Sequential(
        nn.Conv2d(inc,outc,ks,s,p),
        nn.BatchNorm2d(outc),
        nn.ReLU(inplace=True)
        )


class YOLO(nn.Module):

    def __init__(self, class_num):
        super(YOLO, self).__init__()
        
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
        self.layers.append( Layer(256,256,1,1,0))
        self.layers.append( Layer(256,256,3,2,1))
        # self.layers.append( nn.Conv2d(256,4,3,1,1))
        self.boxconv = nn.Conv2d(256,4,3,1,1)
        self.confconv = nn.Conv2d(256,class_num,3,1,1)
       
       

        
                
                


                    

    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        # x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
    
        #TODO: define forward
        for i in range(len(self.layers)):
            x=self.layers[i](x)
         
        

        
        #remember to apply softmax to confidence! Which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,5,5,num_of_classes]
        #bboxes - [batch_size,5,5,4]
        confidence = self.confconv(x)    
        confidence=torch.softmax(confidence,dim=1)
        confidence = confidence.permute(0,2,3,1)
        bboxes = self.boxconv(x)
        bboxes=bboxes.permute(0,2,3,1)
        
        return confidence,bboxes

# class Layer (nn.Module):
#     def __init__(self,inc,outc,ks,s,p):
#         super(Layer, self).__init__()
#         self.conv = nn.Conv2d(inc,outc,ks,s,p)
#         self.bn = nn.BatchNorm2d(outc)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self,x):
#         x = self.conv(x)
#         x=self.bn(x)
#         x=self.relu(x)
#         return x











