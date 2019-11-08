import json
from PIL import Image, ImageDraw
from collections import defaultdict
from tqdm import tqdm

gt = json.load(open('/gruntdata/openImages/ocr/val.json'))
print(gt['images'][0])
id2im_name = {i['id']: i['file_name'] for i in gt['images']}
mp = {1: "TextLineBox", 2: "SubjectBox"}

result = json.load(open('/gruntdata/openImages/ocr/models/faster_rcnn_x101_32x4d_fpn_1x/results.pkl.bbox.json'))
predicted = defaultdict(list)
for entry in result:
    im_name = id2im_name[entry['image_id']]
    category = mp[entry['category_id']]
    conf = entry['score']
    bbox = entry['bbox']
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
    if conf < 0.9:
        continue
    predicted[im_name].append(bbox + [conf, category])

for name, labels in tqdm(predicted.items()):
    im = Image.open('/gruntdata/openImages/ocr/val_images/' + name)
    draw = ImageDraw.Draw(im)
    for label in labels:
        if label[-1] == 'TextLineBox':
            draw.rectangle(label[:4], outline='red')
        elif label[-1] == 'SubjectBox':
            draw.rectangle(label[:4], outline='green')
    im.save('/gruntdata/openImages/ocr/vis/' + name)