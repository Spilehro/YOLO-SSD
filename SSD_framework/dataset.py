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
import os.path
import cv2
import math
import random
# from utils import visualize_pred



#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    num_default_bb = len(large_scale)
    num_cell_size =len(layers)
    size1,size2,size3,size4 = layers
    total_cells = size1**2 + size2**2 + size3**2 + size4**2
    box_num = num_cell_size*total_cells
    boxes = np.zeros((total_cells,num_default_bb,8),dtype=np.float32)

    count = 0
    widths =np.zeros((4,4),dtype=np.float32)
    heights =np.zeros((4,4),dtype=np.float32)

    for i in range(len(widths)):
        w1 = small_scale[i]
        w2 = large_scale[i]
        w3 = min(large_scale[i]*math.sqrt(2),1)
        w4 = min(large_scale[i]/math.sqrt(2),1)
        h1 = small_scale[i]
        h2 = large_scale[i]
        h3 = min(large_scale[i]/math.sqrt(2),1)
        h4 = min(large_scale[i]*math.sqrt(2),1)
        widths[i]=[w1,w2,w3,w4]
        heights[i]=[h1,h2,h3,h4]


    for i in range(len(layers)):
        cell_size = layers[i]
        cell_portion = 1/cell_size
        for j in range(cell_size**2):
            xind = j%cell_size
            yind = math.floor(j/cell_size)
            x_center = xind*cell_portion + cell_portion/2
            y_center = yind*cell_portion + cell_portion/2
            for k in range(4):
                x_min = max(x_center-widths[i][k]/2,0)
                y_min = max(y_center-heights[i][k]/2,0)
                x_max = min(x_center+widths[i][k]/2,1)
                y_max = min(y_center+heights[i][k]/2,1)
                boxes[count][k]=[x_center,y_center,widths[i][k],heights[i][k],x_min,y_min,x_max,y_max]
            count=count+1 

    boxes = boxes.reshape(box_num,8)

    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    gw= x_max-x_min
    gh = y_max-y_min
    gx=x_min+gw/2
    gy = y_min+gh/2
    g=[gx,gy,gw,gh]
    
    ious_true = ious>threshold
    one_hot_vectors = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    if np.any(ious_true):
        indices = np.nonzero(ious_true)
        p= boxs_default[indices][:,0:4]
        # g = ann_box[indices]
        tx =(g[0]-p[:,0])/p[:,2]
        ty =(g[1]-p[:,1])/p[:,3]
        tw =np.log(g[2]/p[:,2])
        th =np.log(g[3]/p[:,3])
        ann_box[indices]=np.transpose([tx,ty,tw,th])
        ann_confidence[indices]=one_hot_vectors[cat_id]
    else:
        ious_true = np.argmax(ious)
        p = boxs_default[ious_true,0:4]
        # g = ann_box[ious_true]
        tx =(g[0]-p[0])/p[2]
        ty =(g[1]-p[1])/p[3]
        tw =np.log(g[2]/p[2])
        th =np.log(g[3]/p[3])
        ann_box[ious_true]=[tx,ty,tw,th]
        ann_confidence[ious_true]=one_hot_vectors[cat_id]
        
    return ann_box,ann_confidence


def get_min_max (ann_file_name):
    gt_box = np.empty((0,5),dtype=np.float32)
    ann = open(ann_file_name)
    x_min = 100000000000
    y_min = 100000000000
    x_max = 0
    y_max = 0

    for line in ann.readlines():

        ann_elems = line.split()
        cat_id = int(ann_elems[0])
        x,y,w,h= float(ann_elems[1]),float(ann_elems[2]),float(ann_elems[3]),float(ann_elems[4])

        gt_box = np.vstack((gt_box,[cat_id,x,y,w,h]))

        x_end = x+w
        y_end = y+h
        x_min = min(x,x_min)
        y_min = min(y,y_min)
        x_max = max(x_end,x_max)
        y_max = max(y_end,y_max)

    ann.close()

    return x_min,y_min,x_max,y_max,gt_box

def randomCrop(image,x_min,y_min,x_max,y_max,gt_box): 

    image_width = image.shape[1]
    image_height = image.shape[0]

    start_x =int(random.uniform(0,x_min))
    start_y =int(random.uniform(0,y_min))

    end_x = int(random.uniform(x_max,image_width))
    end_y = int(random.uniform(y_max,image_height))
    
    image_cropped = image[start_y:end_y,start_x:end_x]

    gt_box[:,1]=gt_box[:,1]-start_x
    gt_box[:,2]=gt_box[:,2]-start_y
    
    return image_cropped,gt_box

def flip_h (image,image_width,gt_box):

    image = cv2.flip(image,1)
    start_x =  gt_box[:,1]
    box_width = gt_box[:,3]
    gt_box[:,1]=image_width-start_x-box_width
    
    return image,gt_box

class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.img_names.sort()
        self.image_size = image_size
        length=len(self.img_names)
        end = round(1*length)

        # if(self.train):
        #     self.img_names = self.img_names[0:end]
        # else:
        #     self.img_names=self.img_names[end:length]
        
        #notice:
        #you can split the dataset into 80% training and 20% testing here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"

        num_augment = 2

        crop_or_not = 0
        # crop_or_not = index%num_augment
        # index = math.floor(index/num_augment)

        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"

        load_train = os.path.exists(ann_name)

        resize_val = self.image_size

        image = cv2.imread(img_name)
        original_size = image.shape

        if load_train:
            x_min,y_min,x_max,y_max,gt_box = get_min_max(ann_name)
            

            #TODO:
            if(crop_or_not):
                image,gt_box = randomCrop(image,x_min,y_min,x_max,y_max,gt_box)
                image_width = image.shape[1]
                image,gt_box = flip_h (image,image_width,gt_box)

            original_size = image.shape

            image = cv2.resize(image,(resize_val,resize_val))
            image = np.swapaxes(image,2,1)
            image = np.swapaxes(image,1,0)
            #image = np.transpose(image)

            for i in range(len(gt_box)):
                cat_id,x,y,width,height = gt_box[i]

                

                x = x/original_size[1]
                y=y/original_size[0]

                
                width = width/original_size[1]
                height = height/original_size[0]

            

                x_min = x
                y_min = y

                

                x_max = x+width 
                y_max = y+height

                cat_id = int(cat_id)

                ann_box,ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,cat_id,x_min,y_min,x_max,y_max)

        
        else:
            image = cv2.resize(image,(resize_val,resize_val))
            image = np.swapaxes(image,2,1)
            image = np.swapaxes(image,1,0)
        image = image/255.0
        image = image.astype("float32")
        
        # boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])
        # visualize_pred("train", ann_confidence, ann_box, ann_confidence, ann_box, image, boxs_default)
        
        return image, ann_box, ann_confidence
