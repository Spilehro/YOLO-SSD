import argparse
import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
args.test = False
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 91
batch_size = 32


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True

data_path = "/scratch/SSD_data/data"
if not args.test:
    dataset = COCO(data_path+"/train/images/", data_path+"/train/annotations/", class_num, boxs_default, train = True, image_size=320)

    dataset_test = COCO(data_path+"/train/images/", data_path+"/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()
    # network.load_state_dict(torch.load('network_10.pth'))
    # network.eval()

    for epoch in range(11,num_epochs):
        #TRAIN
    
        network.train()

        avg_loss = 0
        avg_count = 0
        start_message = "\r epoch "+str(epoch)+" started."
        print(start_message)
        for i, data in enumerate(dataloader, 0):
            # temp = time.time()
            images_, ann_box_, ann_confidence_ = data
            
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            
            pred_confidence, pred_box = network(images)
        
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1
        if(epoch%10==0):
            network_name = 'network_'+str(epoch)+'.pth'
            torch.save(network.state_dict(),network_name)
            # print("\r"+str(avg_count)+" " + str(avg_loss/avg_count), end="")

        print('[%d] time: %f train loss: %f' % (epoch+1, time.time()-start_time, avg_loss/avg_count))
        
        # visualize
        # pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        # pred_box_ = pred_box[0].detach().cpu().numpy()
        # visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)


else:
    #TEST
    dataset_test = COCO(data_path+"/train/images/", data_path+"/train/annotations/", class_num, boxs_default, train = True, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network_90.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        print("\r iter "+str(i),end="")
        images_, ann_box_, ann_confidence_,image_w,image_h = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        
        #pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        image_w = image_w[0].detach().cpu().numpy()
        image_h = image_h[0].detach().cpu().numpy()

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()

        window_name ="train_"+str(i)
        window_name_nms ="train_"+str(i)+"_nms"
        visualize_pred(window_name, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        pred_confidence_nms, pred_box_nms = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        save_txt(i,pred_confidence_nms,pred_box_nms,boxs_default,image_w,image_h)
        visualize_pred(window_name_nms, pred_confidence_nms, pred_box_nms, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
            # cv2.waitKey(1000)


