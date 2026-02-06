import random
import math
import numpy as np
import cv2
import torch
from typing import Tuple, List, Dict

class FlowDataAugmentation:
    """
    Flow 전용 augmentation (Farneback / DenseFlow, 128 == zero encoding).

    - 입력: numpy array H x W x 2, dtype=uint8, (fx+128, fy+128)
    - 출력: output_dict, pos_tuple_list
      output_dict["global_crops"] = [torch.FloatTensor(C,H,W), torch.FloatTensor(C,H,W)]
      각 tensor는 signed flow로 정규화되어 있음: value = (uint8 - 128) / 128.0  => 약 [-1, +1]
    - horizontal flip: fx 채널(채널0)을 부호 반전 시킴 (signed 단계에서)
    """
    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=6,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

    @staticmethod
    def _random_resized_crop_np(img: np.ndarray, size: Tuple[int,int], scale: Tuple[float,float], ratio: Tuple[float,float]=(3/4,4/3), center_crop: bool=False):
        H, W = img.shape[:2]
        area = H * W
        if not center_crop:
            for _ in range(10):
                target_area = area * random.uniform(scale[0], scale[1])
                log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
                aspect_ratio = math.exp(random.uniform(*log_ratio))
                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))
                if w <= W and h <= H and w > 0 and h > 0:
                    i = random.randint(0, H - h)
                    j = random.randint(0, W - w)
                    crop = img[i:i+h, j:j+w]
                    out = cv2.resize(crop, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
                    return out, (i, j, h, w)
        # fallback central crop
        in_ratio = float(W) / float(H)
        if in_ratio < ratio[0]:
            w = W
            h = int(round(w / ratio[0]))
        elif in_ratio > ratio[1]:
            h = H
            w = int(round(h * ratio[1]))
        else:
            w = W
            h = H
        i = (H - h) // 2
        j = (W - w) // 2
        crop = img[i:i+h, j:j+w]
        out = cv2.resize(crop, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        return out, (i, j, h, w)

    @staticmethod
    def _to_tensor_and_normalize_signed(img: np.ndarray) -> torch.FloatTensor:
        """
        img: H x W x C uint8 where encoding is (value + 128)
        Convert to signed float: (uint8 - 128) / 128.0 => approximate [-1, +1]
        Return: torch.FloatTensor C x H x W
        """
        signed = (img.astype('float32') - 128.0) / 128.0
        t = torch.from_numpy(signed).permute(2, 0, 1).contiguous()
        return t

    def __call__(self, frame_name: str, image: np.ndarray, pos_tuple_list=None, center_crop: bool=False):
        """
        image: H x W x 2, uint8 (fx+128, fy+128)
        return: output_dict, global_crops_pos_tuple_list
        """
        output = {}
        output["frame_name"] = frame_name

        do_hflip = random.random() < 0.5

        # global crop 1
        im1_base, g1_pos = self._random_resized_crop_np(image, (self.global_crops_size, self.global_crops_size), self.global_crops_scale, center_crop=center_crop)

        # global crop 2
        if pos_tuple_list is None:
            im2_base, g2_pos = self._random_resized_crop_np(image, (self.global_crops_size, self.global_crops_size), self.global_crops_scale, center_crop=center_crop)
        else:
            p1 = pos_tuple_list[0]
            p2 = pos_tuple_list[1]
            i1, j1, h1, w1 = p1
            i2, j2, h2, w2 = p2
            im1_base = cv2.resize(image[i1:i1+h1, j1:j1+w1], (self.global_crops_size, self.global_crops_size), interpolation=cv2.INTER_LINEAR)
            im2_base = cv2.resize(image[i2:i2+h2, j2:j2+w2], (self.global_crops_size, self.global_crops_size), interpolation=cv2.INTER_LINEAR)
            g1_pos, g2_pos = p1, p2

        # Convert to signed float BEFORE flip (so flip negates fx correctly)
        im1_signed = (im1_base.astype('float32') - 128.0) / 128.0  # H W 2
        im2_signed = (im2_base.astype('float32') - 128.0) / 128.0

        if do_hflip:
            # flip horizontally and negate fx channel (channel 0)
            im1_signed = np.flip(im1_signed, axis=1).copy()
            im1_signed[..., 0] = -im1_signed[..., 0]
            im2_signed = np.flip(im2_signed, axis=1).copy()
            im2_signed[..., 0] = -im2_signed[..., 0]

        g1 = torch.from_numpy(im1_signed).permute(2, 0, 1).contiguous()
        g2 = torch.from_numpy(im2_signed).permute(2, 0, 1).contiguous()

        output["global_crops"] = [g1, g2]
        global_crops_pos_tuple_list = [g1_pos, g2_pos]
        output["global_crops_pos_tuple"] = global_crops_pos_tuple_list

        # local crops
        local_crops = []
        for _ in range(self.local_crops_number):
            c_img, _ = self._random_resized_crop_np(image, (self.local_crops_size, self.local_crops_size), self.local_crops_scale, center_crop=False)
            c_signed = (c_img.astype('float32') - 128.0) / 128.0
            if do_hflip:
                c_signed = np.flip(c_signed, axis=1).copy()
                c_signed[..., 0] = -c_signed[..., 0]
            local_crops.append(torch.from_numpy(c_signed).permute(2, 0, 1).contiguous())

        output["local_crops"] = local_crops
        output["offsets"] = ()
        return output, global_crops_pos_tuple_list