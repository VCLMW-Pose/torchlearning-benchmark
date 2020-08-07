import numpy as np
import torch
import cv2

from .structures import BaseStructure
from .heatmap import Heatmap

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Keypoints(BaseStructure):
    """
    Human pose estimation target.
    """
    def __init__(self, keypoints, size, mode=None):
        super(Keypoints, self).__init__(keypoints)
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        num_keypoints = self.data.shape[0]
        if num_keypoints:
            self.data = self.data.view(num_keypoints, 3)

        self.size = size
        self.mode = mode

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.data.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h

        self.size = size
        self.data = resized_data
        return self

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = type(self).FLIP_INDS
        flipped_data = self.data[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0

        self.data = flipped_data
        return self

    def calibrate(self):
        pass

    def gen_heatmap(self, hms_size, sigma):
        # TODO: Generate heatmap with keypoints
        keypoints = self.data
        num_joints = keypoints.shape[0]
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = keypoints[:, 2]
        target = np.zeros((num_joints, hms_size[1], hms_size[0]),
                          dtype=np.float32)
        tmp_size = sigma * 3
        feat_stride = np.array(self.size) / np.array(hms_size)

        for i in range(num_joints):
            mu_x = int(keypoints[i, 0] / feat_stride[0] + 0.5)
            mu_y = int(keypoints[i, 1] / feat_stride[1] + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= hms_size[0] or ul[1] >= hms_size[1] or br[0] < 0 or br[1] < 0:
                # return image as is
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], hms_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], hms_size[1]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], hms_size[0])
            img_y = max(0, ul[1]), min(br[1], hms_size[1])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, np.expand_dims(target_weight, -1)

    def __getitem__(self, item):
        keypoints = self.data[item]
        return keypoints

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.data))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s

    def drawpose(self, img=None):
        if not img:
            img = np.zeros((5 * self.size[1], 5 * self.size[0], 3), dtype=np.uint8)

        thick = 2
        jointscoor = {}
        if isinstance(self.data, np.ndarray) or isinstance(self.data, torch.Tensor):
            for i in range(len(self.NAMES)):
                jointscoor[self.NAMES[i]] = (int(5 * self.data[i][0]), int(5 * self.data[i][1]))
        else:
            print("Please check argument type!")
            return
        if jointscoor['nose'][0] != -1 and jointscoor['neck'][0] != -1:
            img = cv2.line(img, jointscoor['nose'], jointscoor['neck'], (181, 102, 60), thickness=thick)
        if jointscoor['neck'][0] != -1 and jointscoor['rShoulder'][0] != -1:
            img = cv2.line(img, jointscoor['neck'], jointscoor['rShoulder'], (250, 203, 91), thickness=thick)
        if jointscoor['rShoulder'][0] != -1 and jointscoor['rElbow'][0] != -1:
            img = cv2.line(img, jointscoor['rShoulder'], jointscoor['rElbow'], (35, 198, 77), thickness=thick)
        if jointscoor['rElbow'][0] != -1 and jointscoor['rWrist'][0] != -1:
            img = cv2.line(img, jointscoor['rElbow'], jointscoor['rWrist'], (35, 98, 177), thickness=thick)
        if jointscoor['neck'][0] != -1 and jointscoor['lShoulder'][0] != -1:
            img = cv2.line(img, jointscoor['neck'], jointscoor['lShoulder'], (66, 218, 128), thickness=thick)
        if jointscoor['lShoulder'][0] != -1 and jointscoor['lElbow'][0] != -1:
            img = cv2.line(img, jointscoor['lShoulder'], jointscoor['lElbow'], (62, 121, 58), thickness=thick)
        if jointscoor['lElbow'][0] != -1 and jointscoor['lWrist'][0] != -1:
            img = cv2.line(img, jointscoor['lElbow'], jointscoor['lWrist'], (23, 25, 118), thickness=thick)
        if jointscoor['neck'][0] != -1 and jointscoor['rHip'][0] != -1:
            img = cv2.line(img, jointscoor['neck'], jointscoor['rHip'], (152, 59, 98), thickness=thick)
        if jointscoor['rHip'][0] != -1 and jointscoor['rKnee'][0] != -1:
            img = cv2.line(img, jointscoor['rHip'], jointscoor['rKnee'], (94, 160, 66), thickness=thick)
        if jointscoor['rKnee'][0] != -1 and jointscoor['rAnkle'][0] != -1:
            img = cv2.line(img, jointscoor['rKnee'], jointscoor['rAnkle'], (44, 159, 96), thickness=thick)
        if jointscoor['neck'][0] != -1 and jointscoor['lHip'][0] != -1:
            img = cv2.line(img, jointscoor['neck'], jointscoor['lHip'], (51, 135, 239), thickness=thick)
        if jointscoor['lHip'][0] != -1 and jointscoor['lKnee'][0] != -1:
            img = cv2.line(img, jointscoor['lHip'], jointscoor['lKnee'], (75, 58, 217), thickness=thick)
        if jointscoor['lKnee'][0] != -1 and jointscoor['lAnkle'][0] != -1:
            img = cv2.line(img, jointscoor['lKnee'], jointscoor['lAnkle'], (244, 59, 166), thickness=thick)
        # if jointscoor['nose'][0] != -1 and jointscoor['rEye'][0] != -1:
        #    img = cv2.line(img, jointscoor['nose'], jointscoor['rEye'], (49, 56, 218), thickness=thick)
        # if jointscoor['rEye'][0] != -1 and jointscoor['rEar'][0] != -1:
        #    img = cv2.line(img, jointscoor['rEye'], jointscoor['rEar'], (23, 25, 118), thickness=thick)
        # if jointscoor['nose'][0] != -1 and jointscoor['lEye'][0] != -1:
        #    img = cv2.line(img, jointscoor['nose'], jointscoor['lEye'], (130, 35, 158), thickness=thick)
        # if jointscoor['lEye'][0] != -1 and jointscoor['lEar'][0] != -1:
        #    img = cv2.line(img, jointscoor['lEye'], jointscoor['lEar'], (53, 200, 18), thickness=thick)
        for joint in self.NAMES:
            if jointscoor[joint] != (-1, -1):
                img = cv2.circle(img, jointscoor[joint], 3, (68, 147, 200), -1)
        return img


def _create_flip_indices(names, flip_map):
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


class MWPoseKeypoints(Keypoints):
    NAMES = [
        'nose',
        'neck',
        'rShoulder',
        'rElbow',
        'rWrist',
        'lShoulder',
        'lElbow',
        'lWrist',
        'rHip',
        'rKnee',
        'rAnkle',
        'lHip',
        'lKnee',
        'lAnkle'
    ]
    FLIP_MAP = {
        'rShoulder': 'lShoulder',
        'rElbow': 'lElbow',
        'rWrist': 'lWrist',
        'rHip': 'lHip',
        'rKnee': 'lKnee',
        'rAnkle': 'lAnkle'
    }


# TODO this doesn't look great
MWPoseKeypoints.FLIP_INDS = _create_flip_indices(MWPoseKeypoints.NAMES, MWPoseKeypoints.FLIP_MAP)


def mwpose_kp_connections(keypoints):
    kp_line = [
        [keypoints.index('nose'), keypoints.index('neck')],
        [keypoints.index('rShoulder'), keypoints.index('rElbow')],
        [keypoints.index('rElbow'), keypoints.index('rWrist')],
        [keypoints.index('lShoulder'), keypoints.index('lElbow')],
        [keypoints.index('lElbow'), keypoints.index('lWrist')],
        [keypoints.index('rHip'), keypoints.index('rKnee')],
        [keypoints.index('rKnee'), keypoints.index('rAnkle')],
        [keypoints.index('lHip'), keypoints.index('lKnee')],
        [keypoints.index('lKnee'), keypoints.index('lAnkle')],
        [keypoints.index('rShoulder'), keypoints.index('lShoulder')],
        [keypoints.index('rHip'), keypoints.index('lHip')],
    ]
    return kp_line


MWPoseKeypoints.CONNECTIONS = mwpose_kp_connections(MWPoseKeypoints.NAMES)


class PersonKeypoints(Keypoints):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }


# TODO this doesn't look great
PersonKeypoints.FLIP_INDS = _create_flip_indices(PersonKeypoints.NAMES, PersonKeypoints.FLIP_MAP)


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


PersonKeypoints.CONNECTIONS = kp_connections(PersonKeypoints.NAMES)


# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def keypoints_to_heat_map(keypoints, rois, heatmap_size):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid
