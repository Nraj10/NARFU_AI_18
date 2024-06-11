from dataclasses import dataclass
from threading import Lock, Thread
from module.config import CONFIG
from module.process_queue import ProcessQueue
from .geoparser import CropImage, SourceImage
import os
import numpy as np
from tifffile import imread as tifread
import cv2
import json


class Processor():

    layouts: list[SourceImage]
    index = 0
    task_id = 0
    tasks: dict[int, ]
    lock: Lock
    process_queue: ProcessQueue


    def __init__(self) -> None:
        self.layouts = []
        self.tasks = dict()
        self.lock = Lock()
        layouts_path = os.path.join(CONFIG.data_path, CONFIG.layouts_dir_name)
        for layout_name in os.listdir(layouts_path):
            self.layouts.append(SourceImage(os.path.join(layouts_path, layout_name)))
        self.process_queue = ProcessQueue(self.lock, self.tasks, self.layouts)

    def process_from_array(self, crop_img: np.ndarray, name: str):
        self.task_id += 1
        self.process_queue.add_task(crop_img, name, self.task_id)
        return self.task_id

    def process_from_file(self, path: str,):
        return self.process_from_array(tifread(path), os.path.basename(path))
    
    def get_projectected_img(self, crop: CropImage):
        img = self.layouts[self.index].image
        M, _, matches = crop.match(self.layouts[self.index])
        crop_transformed = cv2.warpPerspective(crop.image, M, (img.shape[0], img.shape[1]))
        mask = np.zeros_like(img, dtype = np.uint8)
        roi_corners = np.int32([
        [
            [0, 0],
            [0, img.shape[0] - 1],
            [img.shape[1] - 1, img.shape[0] - 1],
            [img.shape[1] - 1, 0]
        ]
        ])
        ignore_mask_color = (255,)
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        img2_masked = cv2.bitwise_and(img, mask)
        return cv2.add(img2_masked, crop_transformed)
    
    