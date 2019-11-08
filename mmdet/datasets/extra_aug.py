import mmcv, cv2
import numpy as np
from numpy import random
from glob import glob
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):

    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img
        boxes += np.tile((left, top), 2)
        return img, boxes, labels


class RandomCrop(object):

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return img, boxes, labels

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = (center[:, 0] > patch[0]) * (
                    center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                        center[:, 1] < patch[3])
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                return img, boxes, labels


class Matting(object):
    def __init__(self, path):
        self.normal_imgs = glob(path + '*.jpg')

    def __call__(self, img, boxes, labels):
        height_im, width_im, _ = img.shape
        if np.random.rand() < 0.75:
            normal_im = cv2.imread(np.random.choice(self.normal_imgs))

            if np.random.rand() > 0.5:
                normal_im = normal_im[::-1, :, :]
            if np.random.rand() > 0.5:
                normal_im = normal_im[:, ::-1, :]
            if np.random.rand() > 0.5:
                normal_im = cv2.transpose(normal_im)
                if np.random.rand() > 0.5:
                    normal_im = normal_im[::-1, :, :]
                else:
                    normal_im = normal_im[:, ::-1, :]

            while normal_im.shape[0] < height_im:
                normal_im = np.concatenate([normal_im, normal_im], axis=0)
            while normal_im.shape[1] < width_im:
                normal_im = np.concatenate([normal_im, normal_im], axis=1)
            normal_im = normal_im[:height_im, :width_im]

            for box in boxes:
                # print(normal_im.shape, im.shape)
                x1, y1, x2, y2 = np.array(box, dtype=int)
                # print(box, x1, y1, x2, y2)
                dx1 = np.random.uniform(0, 1)
                dy1 = np.random.uniform(0, 1)
                dx2 = np.random.uniform(0, 1)
                dy2 = np.random.uniform(0, 1)
                x1 -= dx1 * (x2 - x1)
                x2 += dx2 * (x2 - x1)
                y1 -= dy1 * (y2 - y1)
                y2 += dy2 * (y2 - y1)
                x1, y1, x2, y2 = np.round([x1, y1, x2, y2]).astype(int)
                x1 = np.clip(x1, 0, width_im - 1)
                x2 = np.clip(x2, 0, width_im - 1)
                y1 = np.clip(y1, 0, height_im - 1)
                y2 = np.clip(y2, 0, height_im - 1)
                normal_im[y1: y2 + 1, x1: x2 + 1] = img[y1: y2 + 1, x1: x2 + 1]

            img = normal_im
        return img, boxes, labels


class ExtraAugmentation(object):

    def __init__(self,
                 photo_metric_distortion=None,
                 expand=None,
                 random_crop=None,
                 matting=None):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))
        if matting is not None:
            self.transforms.append(Matting(**matting))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels
