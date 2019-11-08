# -*- coding: utf-8 -*-


import numpy as np 
from shapely.geometry import Polygon

def polygon_from_points(points):
        box = np.empty([1, 8], dtype='int32')
        for i in range(len(points)):
            box[0,i] = float(points[i])
        box = box[0].reshape([4, 2])
        return Polygon(box).convex_hull

def get_intersection_area(box1, box2):
    intersection = box1 & box2
    if intersection.area == 0:
        return 0
    else:
        return intersection.area

def get_union_area(box1, box2):
    intersection = get_intersection_area(box1, box2)
    union = box1.area + box2.area - intersection
    return union

def get_IoU(box1, box2):
    IoU = get_intersection_area(box1,box2)/get_union_area(box1, box2)
    return IoU
    
def cal_IoU(bb_groungtruth, bb_prediction):
    #intersection
    IoU = 0
    if len(bb_prediction)==8 & len(bb_groungtruth)==8:        
        poly_prediction = polygon_from_points(bb_prediction)
        poly_groungtruth = polygon_from_points(bb_groungtruth)
        #print(poly_groungtruth)
        IoU = get_IoU(poly_prediction, poly_groungtruth)
    return IoU