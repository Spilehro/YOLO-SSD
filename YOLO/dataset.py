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
import numpy as np
import os
import cv2
# from utilsSARAH import *
import math


def match(ann_box,ann_confidence,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [5,5,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [5,5,number_of_classes], ground truth class labels to be updated
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    size = 5 #the size of the output grid
    
    
    one_hot_vectors = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    # cells =np.arange(cell_portion,1+cell_portion,cell_portion)
    cell_portion = 1/size
    

    width=(x_max-x_min)
    height=(y_max-y_min)

    centerx=x_min+width/2
    centery=y_min+height/2
    
    cell_num_x=math.floor(size*centerx)
    cell_num_y=math.floor(size*centery)

    # for i in range(cells.size):
    #     if(centerx<cells[i]):
    #         cell_num_x = i
    #         break
    # for j in range(cells.size):
    #     if(centery<cells[j]):
    #         cell_num_y = j
    #         break
        
    ann_confidence[cell_num_y][cell_num_x][:] = one_hot_vectors[cat_id]
    
    rel_x = (centerx-cell_portion*cell_num_x)/cell_portion
    rel_y = (centery-cell_portion*cell_num_y)/cell_portion
    width=math.sqrt(width)
    height=math.sqrt(height)

    ann_box[cell_num_y][cell_num_x][0:4] = [rel_x,rel_y,width,height]

    return ann_box,ann_confidence

    
    
    
class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        length=len(self.img_names)
        end = round(0.8*length)

        if(self.train):
            self.img_names = self.img_names[0:end]
        else:
            self.img_names=self.img_names[end:length]
        
        #notice:
        #you can split the dataset into 80% training and 20% testing here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        size = 5 #the size of the output grid
        ann_box = np.zeros([5,5,4], np.float32)#5*5 bounding boxes
        ann_confidence = np.zeros([5,5,self.class_num], np.float32) #5*5 one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,:,-1] = 1 #the default class for all cells is set to "background"
    
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        resize_val = self.image_size
        #TODO:

        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = cv2.imread(img_name)
        original_size = image.shape
        image = cv2.resize(image,(resize_val,resize_val))
        # image = draw_grid(image,pxstep=int(resize_val/size))
        # cv2.imshow("test",image)
        # cv2.waitKey(1)
        image = np.swapaxes(image,2,1)
        image = np.swapaxes(image,1,0)
        #image = np.transpose(image)
    

        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        ann = open(ann_name)
        

        for line in ann.readlines():

            ann_elems = line.split()

            x,y,width,height = float(ann_elems[1]),float(ann_elems[2]),float(ann_elems[3]),float(ann_elems[4])

            x = x/original_size[1]
            y=y/original_size[0]
            width = width/original_size[1]
            height = height/original_size[0]

            x_min = x
            y_min = y

            x_max = x+width 
            y_max = y+height
            cat_id = int(float(ann_elems[0]))
            ann_box,ann_confidence = match(ann_box,ann_confidence,cat_id,x_min,y_min,x_max,y_max)



        ann.close()  

        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        #you may wonder maybe it is better to input [x_center, y_center, box_width, box_height].
        #maybe it is better.
        #BUT please do not change the inputs.
        #Because you will need to input [x_min,y_min,x_max,y_max] for SSD.
        #It is better to keep function inputs consistent.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        image = image/255.0
        image = image.astype("float32")
        
        return image, ann_box, ann_confidence

    
   