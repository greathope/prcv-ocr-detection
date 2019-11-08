# -*- coding: utf-8 -*-

import os
import sys
import check_detection as check
import cal_IoU

#input_dir = sys.argv[1]
#output_dir = sys.argv[2]
#
#truth_dir = os.path.join(input_dir, 'ref')
#submit_dir = os.path.join(input_dir, 'res')

truth_dir = '/gruntdata/openImages/ocr/'
submit_dir = '/gruntdata/openImages/ocr/submit/'
output_dir = '/gruntdata/openImages/ocr/'

gt_files_dir = os.path.join(truth_dir, 'labels') #ground_truth files zip
dt_files_dir = os.path.join(submit_dir, 'answers') #detect results files zip

result_file = os.path.join(output_dir, 'scores.txt')#result file zip

class_indexes = ['0','1']
IoU_thresh_hold = 0.7

#print(os.listdir(input_dir))
# print(os.listdir(truth_dir))
# print(os.listdir(submit_dir))
# print(os.listdir(gt_files_dir))
# print(os.listdir(dt_files_dir))
# print(len(os.listdir(gt_files_dir)))
# print(len(os.listdir(dt_files_dir)))

if not check.check_input_output(gt_files_dir, dt_files_dir, result_file):
    print("Error in input or output files!")
    sys.exit(1)

if not check.check_files_validate(gt_files_dir, dt_files_dir):
    print("Not Enough Files!")
    sys.exit(1)

result_fout = open(result_file, 'w+')

image_list = [x.strip('.json') for x in os.listdir(gt_files_dir)]
gt = {}
dt = {}
for image in image_list:
    gt[image] = check.parse_file(gt_files_dir, image)
    dt[image] = check.parse_file(dt_files_dir, image)
print('Parse Files Done!')
sum_hmean = 0.0
for class_index in class_indexes:
    sum_match = 0
    sum_gt = 0
    sum_dt = 0
    for image in image_list:
        gt_points = gt[image][class_index]
        dt_points = dt[image][class_index]
        gt_match_list = []
        dt_match_list = []
        for gt_num in range(len(gt_points)):
            for dt_num in range(len(dt_points)):
                if (gt_num not in gt_match_list) and (dt_num not in dt_match_list):
                    IoU = cal_IoU.cal_IoU(gt_points[gt_num], dt_points[dt_num])
                    if IoU > IoU_thresh_hold:
                        gt_match_list.append(gt_num)
                        dt_match_list.append(dt_num)
                        sum_match += 1
        sum_gt += len(gt_points)
        sum_dt += len(dt_points)
    precision = 0 if sum_dt==0 else float(sum_match)/sum_dt
    recall = 0 if sum_gt==0 else float(sum_match)/sum_gt
    hmean = 0 if precision+recall==0 else 2.0*precision*recall/(precision+recall)
    if class_index == "0":
        task_label = "text line detection"
    else:
        task_label = "layout detection"
    result_fout.write(task_label + ":" + str(hmean) + '\n')
    print(hmean)
    sum_hmean += hmean
avg_hmean = sum_hmean/float(len(class_indexes))
result_fout.write("hmean:" + str(avg_hmean) + '\n')
print("avg_hmean {}".format(avg_hmean))
result_fout.close()