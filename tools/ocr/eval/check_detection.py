# -*- coding: utf-8 -*-

import os
import json

def check_input_output(gt_files_dir, dt_files_dir, result_file):
    if not os.path.exists(gt_files_dir):
        print('Error in gt!' + gt_files_dir)
        return False
    if not os.path.exists(dt_files_dir):
        print('Error in dt!' + dt_files_dir)
        return False
    result_path = os.path.dirname(result_file)
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    return True

def check_files_validate(gt_files_dir, dt_files_dir):
    len_gt_files = len(os.listdir(gt_files_dir))
    len_dt_files = len(os.listdir(dt_files_dir))
    if not len_gt_files == len_dt_files:
        print('Error in number of dt filed!')
        return False
    return True

def parse_file(files_dir, image, class_indexes=['0','1']):
    file_name = files_dir + '/' + image + '.json'
    fin = open(file_name, 'r')
    info = json.load(fin)
    fin.close()
    image_name = info['image_name']
    image_labels = info['label']
    file_dict = {}
    if not image_name.rstrip('.jpg') == image:
        return
    for label in image_labels:
        box = [int(i) for i in label['location']]
        if label['type'] == 'TextLineBox':
            key = class_indexes[0]
        elif label['type'] == 'SubjectBox':
            key = class_indexes[1]
        if not key in file_dict.keys():
            file_dict[key] = [box]
        else:
            file_dict[key].append(box)
    for class_index in class_indexes:
        if not class_index in file_dict.keys():
            file_dict[class_index] = []
    return file_dict