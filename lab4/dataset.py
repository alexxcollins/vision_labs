import os,sys,re
from pathlib import Path as Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as data
from PIL import Image as PILImage
from torchvision import transforms as transforms
import json
import xml
from xml.dom import minidom
import xml.etree.ElementTree as ET

# this class inherit Pytorch Dataset class
# loads 1 data point:
# 1 image and the vector of labels

class PascalVOC2012DatasetObjectDetection(data.Dataset):

    def __init__(self, **kwargs):
        super(PascalVOC2012DatasetObjectDetection, self).__init__()
        # Classes from Pascal VOC 2012 dataset, in the correct order without the bgr
        self.voc_classes = kwargs['classes']
        # self.dir = kwargs['dir']
        self.dir_label = kwargs['dir_label']
        #self.img_min_size = kwargs['img_min_size']
        # self.imgs = os.listdir(self.dir)
        self.imgs = kwargs['img_list']
        self._classes = kwargs['classes']

    # this method normalizes the image and converts it to Pytorch tensor
    # Here we use pytorch transforms functionality, and Compose them together,
    def transform_img(self, img):
        t_ = transforms.Compose([
                         transforms.ToPILImage(),
                         #transforms.Resize(img_size),
                         transforms.ToTensor(),
                         #transforms.Normalize(mean=[0.407, 0.457, 0.485],
                         #                     std=[0.229,0.224,0.225])])
                         ])
        img = t_(img)
        # returns image tensor (CxHxW)
        return img

     # load one image
     # idx: index in the list of images
    def load_img(self, idx):
        # im = np.array(PILImage.open(os.path.join(self.dir, self.imgs[idx])))
        im = np.array(PILImage.open(self.imgs[idx]))
        im = im[:,:,::-1]
        im = self.transform_img(im)
        return im

     # this method returns the size of the object inside the bounding box:
     # input is a list in format xmin,ymin, xmax,ymax
    def get_size(bbox):
        _h, _w = bbox[3] - bbox[1], bbox[2]-bbox[0]
        size = _h *_w
        return size

     #compute IoU overlap between given bboxes
     # i mage size: HxW
    def get_iou(bbox1, bbox2, img_size): 
        mask = np.zeros(img_size)
        mask[int(bbox1[0]):int(bbox1[2]), int(bbox1[1]):int(bbox1[3])] += 1
        mask[int(bbox2[0]):int(bbox2[2]), int(bbox2[1]):int(bbox2[3])] += 1
        #  if bboxes intersect, there's an area with values > 1
        intersect = np.sum(mask>1)
        union = np.sum(mask>0)
        if intersect>0:
            iou = intersect/union
        else:
            iou = 0
        return iou

     # this returns a lsit of bounding boxes from Cityscapes json file
     # list of classes: list of PASCAL VOC classes

    def extract_bboxes_cityscapes(self, fname, list_of_classes):    
        with open(fname) as f:
            cs = json.load(f)
        objects = cs['objects']
        classes = []
        bboxes = []
        # extract 
        for o in objects:
            if o['label'] in list_of_classes:
                # extract the label 
                classes.append(_classes[o['label']])
                # extract the bounding box from the polygon
                x,y = zip(*o['polygon'])
                min_x, max_x = min(x), max(x)
                min_y, max_y = min(y), max(y)
                bbox = torch.tensor([min_x, min_y, max_x, max_y], dtype=torch.float)         
                bboxes.append(bbox)

        #return a label: class of the object and bbox gt
        label = {}     
        classes = torch.tensor(classes)
        label['labels'] = classes
        label['boxes'] = torch.stack(bboxes)
        return label

     # this returns a list of bounding boxes from Pascal VOC xml file:
     # fname must be an XML file with the name of the class of the object in the bbox,
     # bbox coordinates (xmin, xmax, ymin, ymax) 
     # and a class name 
    def extract_bboxes_pascal(self, idx, class_dict):
        
        classes = []
        bboxes = []
        dims = ['xmin', 'ymin', 'xmax', 'ymax']
        
        fname = Path(self.dir_label)/self.imgs[idx].with_suffix('.xml').name
        root = ET.fromstring(fname.read_text())

        for obj in root.iter('object'):
            name = obj.find('name').text
            # some classes not in Pascal VOC data set (e.g.head)
            if name in class_dict.keys():
                classes.append(class_dict[name])
                # now create bbox list
                bb = []
                bndbox = obj.find('bndbox')
                for dim in dims:
                    bb.append(float(bndbox.find(dim).text))
                bboxes.append(bb)
                
        # output a dictionary with classes and bboxes
        label={}
        label['labels'] = torch.as_tensor(classes, dtype=torch.int64)
        label['boxes'] = torch.as_tensor(bboxes)
        return label       

    def extract_bboxes_pascal3(self, idx, list_of_classes):
        
        fname = Path(self.dir_label)/self.imgs[idx].with_suffix('.xml').name
        classes = []
        bboxes = []
        
        pascal_voc_xml_doc = minidom.parseString(fname.read_text())
        # This returns the list of objects tagged 'name'
        nameval = pascal_voc_xml_doc.getElementsByTagName('name')
        # get the number of total objects in the image
        good_class_ids = []
        
        for id, n in enumerate(nameval):
            name = n.firstChild.nodeValue
            # some classes not in Pascal VOC data set (e.g.head)
            if name in list_of_classes.keys():
                good_class_ids.append(id)                
                classes.append(list_of_classes[name])
                
        coords = ['xmin', 'ymin', 'xmax', 'ymax']
        c_dict = dict(zip(coords, [[],[],[],[],[]]))
        for c in coords:
            val = pascal_voc_xml_doc.getElementsByTagName(c)
            for id in good_class_ids:
                m = float(val[id].firstChild.nodeValue)
                c_dict[c].append(m)
        
        # put the coordinates together
        for i in range(len(good_class_ids)):
            obj = [c_dict['xmin'][i], c_dict['ymin'][i],
                   c_dict['xmax'][i], c_dict['ymax'][i]]
            bboxes.append(obj)
            
        # output a dictionary with classes and bboxes
        label={}
        label['labels'] = torch.as_tensor(classes, dtype=torch.int64)
        label['boxes'] = torch.as_tensor(bboxes)
        return label


    
    
    def extract_bboxes_pascal2(self, idx, list_of_classes):
        # fname = self.imgs[idx].split('.')[0] + '.xml'
        fname = self.imgs[idx].with_suffix('.xml').name
        fname = os.path.join(self.dir_label, fname)
        classes = []
        bboxes = []
        pascal_voc_xml_doc = minidom.parseString(fname.read_text())
        # This returns the list of objects tagged 'name'
        nameval = pascal_voc_xml_doc.getElementsByTagName('name')
        # get the number of total objects in the image
        total_objects = len(nameval)
        wrong_class_ids = []
        good_class_id = []
        for id, n in enumerate(nameval):
            name = n.firstChild.nodeValue
            # some classes not in Pascal VOC data set (e.g.head)
            if name not in list_of_classes.keys():
                total_objects -=1
                wrong_class_ids.append(id) 
            else:
                good_class_id.append(id)                
                classes.append(list_of_classes[name])

        xminvals = []
        yminvals = []
        xmaxvals = []
        ymaxvals = []
        # this returnx xmin, ymin, xmax and ymax values of the bbox
        xminval = pascal_voc_xml_doc.getElementsByTagName('xmin')
        yminval = pascal_voc_xml_doc.getElementsByTagName('ymin')
        xmaxval = pascal_voc_xml_doc.getElementsByTagName('xmax')
        ymaxval = pascal_voc_xml_doc.getElementsByTagName('ymax')
        # get xmins for all objects
        for idx, xmincode in enumerate(xminval):
            xmin = float(xmincode.firstChild.nodeValue)
            if idx in good_class_id:
                xminvals.append(xmin)

        # get ymins for all objects
        for idx, ymincode in enumerate(yminval):
            ymin = float(ymincode.firstChild.nodeValue)
            if idx in good_class_id:
                yminvals.append(ymin)

        # get xmaxs for all objects
        for idx, xmaxcode in enumerate(xmaxval):
            xmax = float(xmaxcode.firstChild.nodeValue)
            if idx in good_class_id:
                xmaxvals.append(xmax)

        # get ymaxs for all objects
        for idx, ymaxcode in enumerate(ymaxval):
            ymax = float(ymaxcode.firstChild.nodeValue)
            if idx in good_class_id:
                ymaxvals.append(ymax)

        # put the coordinates together
        for i in range(total_objects):
            obj = [xminvals[i], yminvals[i],xmaxvals[i],ymaxvals[i]]
            bboxes.append(obj)
        # output a dictionary with classes and bboxes
        label={}
        label['labels'] = torch.as_tensor(classes, dtype=torch.int64)
        label['boxes'] = torch.as_tensor(bboxes)
        return label

    #'magic' method: size of the dataset
    def __len__(self):
        # return len(os.listdir(self.dir))
        return(len(self.imgs))

     #'magic' method: iterates through the dataset directory to return the image and its gt
    def __getitem__(self, idx):
        # here you have to implement functionality using the methods in this class to return X (image) and y (its label)
        X = self.load_img(idx)        
        y = self.extract_bboxes_pascal(idx, self._classes) 
        return idx, X,y

