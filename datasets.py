import os
import csv
import torch
import pickle
import random
import numpy as np

from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import xml.etree.ElementTree as ET
from torchvision import datasets as datasets


class CUBE(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path: str, 
        split: str = "train",
        certainty: str = 'definitely',
        multi=True,
        transform = None,
        target_transform=None,
        boxcrop=None
        ) -> None:
        self.data_path = data_path
        self.split = split
        self.certainty = certainty
        self.boxcrop = boxcrop
        self.multi = multi
        if not os.path.exists(data_path):
            raise RuntimeError("Folder with data for IMaterialist not exists")
        
        if split == "train":
            file_load = os.path.join(self.data_path, "train.pkl")
        elif split == "val":
            file_load = os.path.join(self.data_path, "val.pkl")
        self.file_names, self.labels = self._load_data(file_load)        
        self.boxes = self._load_boxes(os.path.join(data_path, "bounding_boxes.txt"))
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_data(self, path: str):
        row_class_dict = {}
        row_filename_dict = {}
        if self.certainty == "definitely":
            certainty_keep = [4]
        elif self.certainty == "probably":
            certainty_keep = [4, 3]
        elif self.certainty == "guessing":
            certainty_keep = [4, 3, 2]
        elif self.certainty == "not visible":
            certainty_keep = [4, 3, 2, 1]
        with open(path, 'rb') as file:
            file_list = pickle.load(file)
            for file_info in file_list:
                image_id = file_info['id']
                img_full_path = file_info['img_path']
                img_path_dir = img_full_path.split('/')
                img_path = os.path.join(*img_path_dir[-3:])
                if self.multi:
                    attrs = file_info['attribute_label']
                    centrs = file_info['attribute_certainty']
                    for idx in range(len(attrs)):
                        if attrs[idx] and centrs[idx] not in certainty_keep:
                            attrs[idx] = 0
                    if sum(attrs) > 0:
                        row_filename_dict[image_id] = img_path
                        row_class_dict[image_id] = attrs
                else:
                    row_filename_dict[image_id] = img_path
                    row_class_dict[image_id] = [0 for _ in range(200)] 
                    row_class_dict[image_id][file_info['class_label']] = 1

        return row_filename_dict, row_class_dict
    
    def _load_boxes(self, path: str):
        row_data_dict = {}
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line_list = line.split(' ')
                if int(line_list[0]) in list(self.labels.keys()):
                    x1 = float(line_list[1])
                    y1 = float(line_list[2])
                    x2 = x1 + float(line_list[3])
                    y2 = y1 + float(line_list[4])
                    row_data_dict[int(line_list[0])] = [[x1,y1,x2,y2]]
                    
        return row_data_dict

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        image_id = list(self.file_names.keys())[index]
        image = Image.open(os.path.join(self.data_path, self.file_names[image_id])).convert('RGB')
        label = self.labels[image_id]
        if self.multi:
            target = torch.zeros((3, 112), dtype=torch.long)
        else:
            target = torch.zeros((3, 200), dtype=torch.long)
        target[0] = torch.tensor(label, dtype=torch.long)
        
        if self.boxcrop:
            image = crop_box(image, self.boxes[image_id], self.boxcrop, cut_img=0.2)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
    
    
class MultiLabelNUS(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path: str, 
        split: str = "train",
        transform = None,
        target_transform=None,
        ) -> None:
        self.data_path = data_path
        self.split = split
        if not os.path.exists(data_path):
            raise RuntimeError("Folder with data for IMaterialist not exists")
        
        self.classes_names = []
        with open(os.path.join(data_path, "Concepts81.txt")) as f:
            lines = f.readlines()
            for line in lines:
                self.classes_names.append(line.strip())
                
        self.labels = self._load_labels(os.path.join(data_path, "annotations.csv"))
        self.file_names = list(self.labels.keys())
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_labels(self, path: str):
        row_data_dict = {}
        with open(path, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                if self.split == row[2].strip():
                    classes = eval(row[1])
                    row_data_dict[row[0]] = [0 for _ in range(81)]
                    for class_in in classes:
                        row_data_dict[row[0]][self.classes_names.index(class_in)] = 1
        return row_data_dict
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_path, self.file_names[index])).convert('RGB')
        label = self.labels[self.file_names[index]]
        target = torch.zeros((3, 81), dtype=torch.long)
        target[0] = torch.tensor(label, dtype=torch.long)
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
    
    
class MultiLabelCelebA(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path: str, 
        split: str = "train",
        transform = None,
        target_transform=None
        ) -> None:
        self.data_path = data_path
        self.split = split
        if not os.path.exists(data_path):
            raise RuntimeError("Folder with data for CELEBA not exists")
        
        keep = []
        with open(os.path.join(data_path, "list_eval_partition.txt")) as f:
            lines = f.readlines()
            for line in lines:
                filename, numb = line.split()
                if split == "train" and int(numb) == 0:
                    keep.append(filename)
                elif split == "valid" and int(numb) == 1:
                    keep.append(filename)
                elif split == "test" and int(numb) == 2:
                    keep.append(filename)
                elif split == "all":
                    keep.append(filename)
        
        self.label_names, self.labels = self._load_labels(os.path.join(data_path, "list_attr_celeba.txt"), keep)
        self.file_names = list(self.labels.keys())
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_labels(self, path: str, keep = None):
        row_names = []
        row_data_dict = {}
        with open(path, 'r') as file:
            reader = csv.reader(file)
            line_numb = 0
            for row in reader:
                row_data = row[0]
                if line_numb == 1:
                    row_names = row_data.split()
                elif line_numb > 1:
                    row_data_numb = row_data.split()
                    filename = row_data_numb[0]
                    if keep is not None:
                        if filename in keep:
                            row_data_values = [int(x) if int(x) == 1 else int(x)+1 for x in row_data_numb[1:]]
                            row_data_dict[filename] = row_data_values
                    else:
                        row_data_values = [int(x) if int(x) == 1 else int(x)+1 for x in row_data_numb[1:]]
                        row_data_dict[filename] = row_data_values

                line_numb += 1
                
        return row_names, row_data_dict
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.data_path, "img_align_celeba", self.file_names[index])).convert('RGB')
        label = self.labels[self.file_names[index]]
        target = torch.zeros((3, 40), dtype=torch.long)
        target[0] = torch.tensor(label, dtype=torch.long)
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class VOCDataset(datasets.coco.CocoDetection):
    def __init__(self, root, transform=None, target_transform=None, val=False, boxcrop=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.boxcrop = boxcrop
        if val:
            image_sets_file = os.path.join(self.root,"ImageSets/Main/train.txt")
        else:
            image_sets_file = os.path.join(self.root,"ImageSets/Main/val.txt")
        self.ids = VOCDataset._read_image_ids(image_sets_file)

        # if the labels file exists, read in the class names
        self.class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        target = torch.zeros((3, len(self.class_names)), dtype=torch.long)
        target[0] = torch.tensor(labels, dtype=torch.long)
        img = Image.open(os.path.join(self.root, f"JPEGImages/{image_id}.jpg")).convert('RGB')
        
        if self.boxcrop:
            img = crop_box(img, boxes, self.boxcrop, cut_img=0.2)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.root, f"Annotations/{image_id}.xml")
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = [0 for x in range(len(self.class_names))]
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels[self.class_dict[class_name]] = 1
        return boxes, labels
    
    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, boxcrop=None):
        self.root = root
        self.coco = COCO(annFile)
        self.boxcrop = boxcrop
        
        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        boxes = [x['bbox'] for x in target]

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.boxcrop:
            img = crop_box(img, boxes, self.boxcrop, cut_img=0.15)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
        
        return x


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def crop_box(img, boxes, size, scale=(0.2, 0.2), cut_img=0.1):
    width, height = img.size
    max_height = height * scale[0]
    max_width = width * scale[1]
    crop_height_up = float(torch.rand(1) * max_height)
    crop_width_left = float(torch.rand(1) * max_width)
    crop_height_down = float(torch.rand(1) * max_height)
    crop_width_right = float(torch.rand(1) * max_width)
    for box in boxes:
        x1, y1, x2, y2 = box
        if crop_width_left > x1:
            if crop_width_left-x1 > (x2-x1)*cut_img:
                crop_width_left = x1 + (x2-x1)*cut_img
        if width-crop_width_right < x2:
            if x2-(width-crop_width_right) > (x2-x1)*cut_img:
                crop_width_right = width - (x2-(x2-x1)*cut_img)
        if height-crop_height_down < y2:
            if y2-(height-crop_height_down) > (y2-y1)*cut_img:
                crop_height_down = height - (y2-(y2-y1)*cut_img)
        if crop_height_up > y1:
            if crop_height_up-y1 > (y2-y1)*cut_img:
                crop_height_up = y1 + (y2-y1)*cut_img

    cropped_img = img.crop((int(crop_width_left),int(crop_height_up),int(width-crop_width_right),int(height-crop_height_down)))
    resized_img = cropped_img.resize((size, size))
    return resized_img