import pickle
import cv2
import os
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)   

def serialize_json(obj, path):
    with open(path, 'w') as fh:
        json.dump(obj, fh)
 
def deserialize_json(path):
    with open(path, 'r') as fh:
        return json.load(fh)

def convert_video2imgs(video_read_path, every_frame, img_save_path):
    cam = cv2.VideoCapture(video_read_path)
    curr_frame = 0
    print("Saving video as images: sampling every " + str(every_frame) + " frames")
    while(True):
        ret,frame = cam.read()    
        if ret: # if video frames remain
            if (curr_frame % every_frame) == 0:
                save_name = 'frame' + str(curr_frame) + '.jpg'
                save_path = os.path.join(img_save_path, save_name)
                print('Creating...' + save_name)
                cv2.imwrite(save_path, frame)
            curr_frame += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()

#adapted from: https://machinelearningspace.com/coco-dataset-a-step-by-step-guide-to-loading-and-visualizing/
class COCOParser:
    def __init__(self, anns_file, imgs_dir):
        with open(anns_file, 'r') as f:
            coco = json.load(f)
            
        self.annIm_dict = defaultdict(list)        
        self.cat_dict = {} 
        self.annId_dict = {}
        self.im_dict = {}
        self.licenses_dict = {}
        for ann in coco['annotations']:           
            self.annIm_dict[ann['image_id']].append(ann) 
            self.annId_dict[ann['id']]=ann
        for img in coco['images']:
            self.im_dict[img['id']] = img
        for cat in coco['categories']:
            self.cat_dict[cat['id']] = cat
        for license in coco['licenses']:
            self.licenses_dict[license['id']] = license
    def get_imgIds(self):
        return list(self.im_dict.keys())
    def get_annIds(self, im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        return [ann['id'] for im_id in im_ids for ann in self.annIm_dict[im_id]]
    def load_anns(self, ann_ids):
        im_ids=ann_ids if isinstance(ann_ids, list) else [ann_ids]
        return [self.annId_dict[ann_id] for ann_id in ann_ids]        
    def load_cats(self, class_ids):
        class_ids=class_ids if isinstance(class_ids, list) else [class_ids]
        return [self.cat_dict[class_id] for class_id in class_ids]
    def get_imgLicenses(self,im_ids):
        im_ids=im_ids if isinstance(im_ids, list) else [im_ids]
        lic_ids = [self.im_dict[im_id]["license"] for im_id in im_ids]
        return [self.licenses_dict[lic_id] for lic_id in lic_ids]
    def map_filename2imgid(self):
        new_dict = {}
        for im_id in self.im_dict.keys():
            filename = self.im_dict[im_id]["file_name"]
            new_dict[filename] = im_id
        return new_dict

def plot_random_coco(coco, coco_images_dir):
    # define a list of colors for drawing bounding boxes
    color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta","green","aqua"]*10
    num_imgs_to_disp = 4
    total_images = len(coco.get_imgIds()) # total number of images
    sel_im_idxs = np.random.permutation(total_images)[:num_imgs_to_disp]
    img_ids = coco.get_imgIds()
    selected_img_ids = [img_ids[i] for i in sel_im_idxs]
    ann_ids = coco.get_annIds(selected_img_ids)
    print("image ids:", sel_im_idxs)
    print("annotation ids:", ann_ids)
    im_licenses = coco.get_imgLicenses(selected_img_ids)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    ax = ax.ravel()

    for i, im in enumerate(selected_img_ids):
        image = Image.open(f"{coco_images_dir}/{str(im).zfill(12)}.jpg")
        ann_ids = coco.get_annIds(im)
        annotations = coco.load_anns(ann_ids)
        for ann in annotations:
            bbox = ann['bbox']
            x, y, w, h = [int(b) for b in bbox]
            class_id = ann["category_id"]
            class_name = coco.load_cats(class_id)[0]["name"]
            license = coco.get_imgLicenses(im)[0]["name"]
            color_ = color_list[class_id]
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_, facecolor='none')
            t_box=ax[i].text(x, y, class_name,  color='red', fontsize=10)
            t_box.set_bbox(dict(boxstyle='square, pad=0',facecolor='white', alpha=0.6, edgecolor='blue'))
            ax[i].add_patch(rect)
        
        ax[i].axis('off')
        ax[i].imshow(image)
        ax[i].set_xlabel('Longitude')
        ax[i].set_title(f"License: {license}")
    plt.tight_layout()
    plt.show()

def plot_coco_image(coco, coco_images_dir, img_path):
    # define a list of colors for drawing bounding boxes
    color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta","green","aqua"]*10
    img_ids = coco.get_imgIds()
    mapper = coco.map_filename2imgid()
    selected_img_ids = [mapper[img_path.split("/")[-1]]]
    print(selected_img_ids)
    ann_ids = coco.get_annIds(selected_img_ids)
    im_licenses = coco.get_imgLicenses(selected_img_ids)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))

    im = selected_img_ids[0]
    image = Image.open(f"{coco_images_dir}/{str(im).zfill(12)}.jpg")
    ann_ids = coco.get_annIds(im)
    annotations = coco.load_anns(ann_ids)
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(b) for b in bbox]
        class_id = ann["category_id"]
        class_name = coco.load_cats(class_id)[0]["name"]
        license = coco.get_imgLicenses(im)[0]["name"]
        color_ = color_list[class_id]
        rect = plt.Rectangle((x, y), w, h, linewidth=4, edgecolor=color_, facecolor='none')
        t_box=ax.text(x, y, class_name,  color='red', fontsize=20)
        t_box.set_bbox(dict(boxstyle='square, pad=0',facecolor='white', alpha=0.6, edgecolor='blue'))
        ax.add_patch(rect)
    
    ax.axis('off')
    ax.imshow(image)
    ax.set_xlabel('Longitude')
    # ax.set_title(f"License: {license}")
    plt.tight_layout()
    plt.show()

def get_coco_labels(coco, coco_images_dir, img_path):
    # define a list of colors for drawing bounding boxes
    img_ids = coco.get_imgIds()
    mapper = coco.map_filename2imgid()
    selected_img_ids = [mapper[img_path.split("/")[-1]]]
    ann_ids = coco.get_annIds(selected_img_ids)

    im = selected_img_ids[0]
    image = Image.open(f"{coco_images_dir}/{str(im).zfill(12)}.jpg")
    im_w, im_h = image.size
    pixels = im_w * im_h

    ann_ids = coco.get_annIds(im)
    annotations = coco.load_anns(ann_ids)

    counts_dict = {}
    size_dict = {}
    # load with 0s
    for class_id in list(coco.cat_dict.keys()):
        class_name = coco.load_cats(class_id)[0]["name"]
        counts_dict[class_name] = 0
        size_dict[class_name] = 0
    # count up occurrences
    for ann in annotations:
        bbox = ann['bbox']
        x, y, w, h = [int(b) for b in bbox]
        class_id = ann["category_id"]
        class_name = coco.load_cats(class_id)[0]["name"]
        counts_dict[class_name] += 1
        size_dict[class_name] += (w*h) / pixels
    return counts_dict, size_dict

def process_imagenet_labels(label_path):
    #labels extracted from: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    label_list = []
    with open(label_path, 'r') as file:
        for line in file:
            mod_line = str(line.split(":")[1].rstrip().split("'")[1])
            label_list.append(mod_line)
    return label_list

def process_coco_labels(label_path):
    label_list = []
    with open(label_path, 'r') as file:
        for line in file:
            mod_line = str(line.rstrip())
            label_list.append(mod_line)
    return label_list

# def plot_seg_predictions():
    #         rect = plt.Rectangle((x, y), w, h, linewidth=4, edgecolor=color_, facecolor='none')
    #     t_box=ax.text(x, y, class_name,  color='red', fontsize=20)
    #     t_box.set_bbox(dict(boxstyle='square, pad=0',facecolor='white', alpha=0.6, edgecolor='blue'))
    #     ax.add_patch(rect)
    
    # ax.axis('off')
    # ax.imshow(image)
    # ax.set_xlabel('Longitude')
    # # ax.set_title(f"License: {license}")
    # plt.tight_layout()
    # plt.show()

def ndcg_simple(relevscore,k,gtlength):
    dcg=0.0
   # print(k,len(relevscore))
    kmin=min(k,len(relevscore))
    ideal=0
    for i in range(kmin):
        val=relevscore[i]
        weight=1/math.log(i+2, 2)
        #dcg+=val*weight
        if (i<len(relevscore)):
            dcg+=val*weight
        else:
            dcg+=0*weight
        if (i<gtlength):
            ideal+=1*weight #since relev score=1
        else:
            ideal+=0*weight 
       # print(val,weight)
    #dcg/=kmin
    #ideal/=kmin
    #print(dcg,ideal)
    ndcg=dcg/ideal
    return ndcg