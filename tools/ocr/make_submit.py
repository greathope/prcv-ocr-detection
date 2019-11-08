import json
from collections import defaultdict
from tqdm import tqdm
import os

gt = json.load(open('/gruntdata/openImages/ocr/val.json'))
print(gt['images'][0])
id2im_name = {i['id']: i['file_name'] for i in gt['images']}
mp = {1: "TextLineBox", 2: "SubjectBox"}

# entries = json.load(open('/gruntdata/openImages/ocr/models/faster_rcnn_x101_32x4d_fpn_1x/results.pkl.bbox.json'))
# entries = json.load(open('/gruntdata/openImages/ocr/models/cascade_rcnn_x101_32x4d_fpn_1x/results.pkl.bbox.json'))
# entries = json.load(open('/gruntdata/openImages/ocr/models/cascade_rcnn_dcn_x101_32x4d_fpn_1x/results.pkl.bbox.json'))
# entries = json.load(open('/grunt/hope/others/ocr/test/ocr_val/generalized_rcnn/bbox_ocr_val_results.json'))
# entries = json.load(open('/gruntdata/openImages/ocr/models/cascade_rcnn_dcn_x101_64x4d_fpn_1x/results.pkl.bbox.json'))
entries = json.load(open('/gruntdata/openImages/ocr/models/cascade_rcnn_dcn_x101_32x4d_fpn_1x_trainval/results.pkl.bbox.json'))
predicted = defaultdict(list)
for entry in entries:
    im_name = id2im_name[entry['image_id']]
    category = mp[entry['category_id']]
    conf = entry['score']
    bbox = entry['bbox']
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    if conf < 0.5:
        continue
    predicted[im_name].append(bbox + [conf, category])

counter = 0
for name, labels in tqdm(predicted.items()):
    result = {}
    out = []
    for label in labels:
        out.append({
            "content": "",
            'type': label[-1],
            'location': [label[0], label[1], label[2], label[1], label[2], label[3], label[0], label[3]]
        })
        counter += 1
    result['image_name'] = name
    result['label'] = out
    json.dump(result, open('/gruntdata/openImages/ocr/submit/answers/' + name.split('.')[0] + '.json', 'w'))
# os.system("cd /gruntdata/openImages/ocr/submit/; zip -r answers.zip answers")
print("{} boxes".format(counter))
print("Writing to /gruntdata/openImages/ocr/submit") #/answers.zip")
os.system('python eval/cal_hmean.py')
# 0.9 0.6754 6529
# 0.95 0.6526967198  6169
# 0.85 6703

#### CASCADE ####
# 0.9 6239 0.688
# 0.85 6467 0.7069429582
# 0.8 6608 0.7121319398

