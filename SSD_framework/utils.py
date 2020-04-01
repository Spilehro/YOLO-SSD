import numpy as np
import cv2
from dataset import iou
import math
import os
import os.path


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use red green blue to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    image_ = image_*255
    image_size = image_.shape[1]
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    threshold=0.7
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        px,py,pw,ph,px_min,py_min,px_max,py_max=boxs_default[i]
        for j in range(class_num):
            if ann_confidence[i,j]>threshold: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                dx,dy,dw,dh = ann_box[i,0:4]
                gx = pw*dx+px
                gy = ph*dy+py
                gw = pw*np.exp(dw)
                gh = ph*np.exp(dh)
                #you can use cv2.rectangle as follows:
                gx_min = int((gx-gw/2)*image_size)
                gy_min =int((gy-gh/2)*image_size)
                gx_max=int((gx+gw/2)*image_size)
                gy_max=int((gy+gh/2)*image_size)
                start_point_gt = (gx_min, gy_min) 
                end_point_gt = (gx_max,gy_max) #bottom right corner
                start_point_default = (int(px_min*image_size), int(py_min*image_size)) 
                end_point_default = (int(px_max*image_size), int(py_max*image_size))
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                cv2.rectangle(image1, start_point_gt, end_point_gt, color, thickness)
                cv2.rectangle(image2, start_point_default, end_point_default, color, thickness)
    
    #pred
    for i in range(len(pred_confidence)):
        px,py,pw,ph,px_min,py_min,px_max,py_max=boxs_default[i]
        for j in range(class_num):
            if pred_confidence[i,j]>threshold:
                dx,dy,dw,dh = pred_box[i]
                gx = pw*dx+px
                gy = ph*dy+py
                gw = pw*np.exp(dw)
                gh = ph*np.exp(dh)
                #you can use cv2.rectangle as follows:
                gx_min = int((gx-gw/2)*image_size)
                gy_min =int((gy-gh/2)*image_size)
                gx_max=int((gx+gw/2)*image_size)
                gy_max=int((gy+gh/2)*image_size)
                start_point_gt = (gx_min, gy_min) 
                end_point_gt = (gx_max,gy_max) #bottom right corner
                start_point_default = (int(px_min*image_size), int(py_min*image_size)) 
                end_point_default = (int(px_max*image_size), int(py_max*image_size))
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                cv2.rectangle(image3, start_point_gt, end_point_gt, color, thickness)
                cv2.rectangle(image4, start_point_default, end_point_default, color, thickness)
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    file_dir = 'data/results_train/'
    # test=os.path.exists(file_dir)
    file_name = file_dir +windowname+".jpg"
    cv2.imwrite(file_name,image)
    # cv2.imshow("augment_check",image)
    # cv2.waitKey()
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.3, threshold=0.7):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    num_boxes = confidence_.shape[0]
    nms_boxes = np.zeros((num_boxes,4),dtype=np.float32)
    nms_confidence = np.zeros((num_boxes,4),dtype=np.float32)
    nms_confidence[:,-1]=1
    

    px = boxs_default[:,0]
    py = boxs_default[:,1]
    pw = boxs_default[:,2]
    ph = boxs_default[:,3]

    dx = box_[:,0]
    dy = box_[:,1]
    dw = box_[:,2]
    dh = box_[:,3]

    g=np.zeros([len(dx),8],dtype=np.float32)
    gx = pw*dx+px
    gy = ph*dy+py
    gw = pw*np.exp(dw)
    gh = ph*np.exp(dh)
    gx_min =gx-gw/2
    
    gy_min = gy-gh/2
   
    gx_max =gx+gw/2
  
    gy_max = gy+gh/2
  
    g[:,0]=gx
    g[:,1]=gy
    g[:,2]=gw
    g[:,3]=gh
    g[:,4]=gx_min
    g[:,5]=gy_min
    g[:,6]=gx_max
    g[:,7]=gy_max

    

    while True:
        index = np.unravel_index(np.argmax(confidence_[:,0:3]),confidence_[:,0:3].shape)
        highest_conf_index =index[0]
        cat_id = index[1]
        if confidence_[highest_conf_index,cat_id] <= threshold:
            break
        else:
            x=box_[highest_conf_index]

            nms_boxes[highest_conf_index]=x
            nms_confidence[highest_conf_index]=confidence_[highest_conf_index]

            _,_,_,_,x_min,y_min,x_max,y_max = g[highest_conf_index]

            confidence_[highest_conf_index]=[0.0,0.0,0.0,1.0]
            g[highest_conf_index]=[0,0,0,0,0,0,0,0]
            
            
            same_cat_index = np.nonzero(confidence_[:,cat_id]>threshold)[0]
             
            ious_true =np.nonzero(iou(g[same_cat_index,:], x_min,y_min,x_max,y_max)>overlap)[0]
            remove_indices = same_cat_index[ious_true]

            g[remove_indices]=[0,0,0,0,0,0,0,0]
            confidence_[remove_indices]=[0.0,0.0,0.0,1.0]
            


    # print('finished')
    return nms_confidence ,nms_boxes
    
    #TODO: non maximum suppression
def save_txt (file_index,pred_confidence,pred_box,boxs_default,image_w,image_h,threshold=0.7,image_size=320):

    file_dir = 'data/texts_train/'
    file_name = str(file_index)
    file_name=file_name.rjust(5, '0')
    file_name = file_dir+file_name+'.txt'
    f= open(file_name,"w+")
    
    class_num = pred_confidence.shape[1]-1

    for i in range(len(pred_confidence)):
        px,py,pw,ph=boxs_default[i,0:4]
        for j in range(class_num):
            if pred_confidence[i,j]>threshold:

                dx,dy,dw,dh = pred_box[i]

                gx = pw*dx+px
                gy = ph*dy+py

                gw = round(pw*np.exp(dw)*image_w,2)
                gh = round(ph*np.exp(dh)*image_h,2)

                gx_min = round(gx*image_w-gw/2,2)
                gy_min =round(gy*image_h-gh/2,2)

                cat_id = j

                line = str(cat_id)+" "+str(gx_min)+" "+str(gy_min)+" "+str(gw)+" "+str(gh)+"\n"
                f.write(line)
    f.close()





