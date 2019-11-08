import json
from tqdm import tqdm
import cv2, os
from glob import glob
from collections import defaultdict

base_dir = '/gruntdata/openImages/ocr/OCR-train-publish-v2/images_labels/'
# base_dir = '/gruntdata/openImages/ocr/val/'
mp = {"TextLineBox":1, "SubjectBox": 2}

def make_coco_traindataset(im_paths, mode='train'):

    idx = 1
    image_id = 20190000000
    images = []
    annotations = []

    for im_path in tqdm(im_paths):

        im_name = os.path.basename(im_path)
        im = cv2.imread(im_path)
        h, w, _ = im.shape
        image_id += 1
        image = {'file_name': im_name, 'width': w, 'height': h, 'id': image_id}
        images.append(image)

        annos = json.load(open(base_dir + 'labels/' + im_name.replace('jpg', 'json')))['label']
        for anno in annos:
            seg = anno['location']
            bbox = [seg[0], seg[1], seg[4] - seg[0], seg[5] - seg[1]]
            anno_ = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': mp[anno['type']], 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(anno_)

    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations
    category = [{'supercategory':'none', 'id': id, 'name': name} for name, id in mp.items()]
    ann['categories'] = category
    # json.dump(ann, open('/gruntdata/openImages/ocr/OCR-train-publish-v2/{}.json'.format(mode),'w'))
    json.dump(ann, open('/gruntdata/openImages/ocr/val/{}.json'.format(mode), 'w'))

img_paths = glob(base_dir + 'val_images/*.jpg')
print(len(img_paths))
make_coco_traindataset(img_paths, 'val')

# make_coco_testdataset()


def make_coco_testdataset():
    img_paths = glob('/gruntdata/openImages/ocr/val_images/*.jpg')
    # assert len(img_paths) == 1000

    idx = 1
    image_id = 20190000000
    images = []
    annotations = []
    for img_path in tqdm(img_paths):

        im = cv2.imread(img_path)
        h, w, _ = im.shape
        image_id += 1
        image = {'file_name': os.path.basename(img_path), 'width': w, 'height': h, 'id': image_id}
        images.append(image)

        anno_ = {'segmentation': [[]], 'area': 100, 'iscrowd': 0, 'image_id': image_id,
               'bbox': [10, 10, 10, 10], 'category_id': 1, 'id': idx, 'ignore': 0}
        idx += 1
        annotations.append(anno_)

    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations
    category = [{'supercategory':'none', 'id': id, 'name': name} for name, id in mp.items()]
    ann['categories'] = category
    json.dump(ann, open('/gruntdata/openImages/ocr/val.json','w'))


# make_coco_testdataset()
