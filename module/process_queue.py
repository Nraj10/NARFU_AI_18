from dataclasses import dataclass
from datetime import date
from queue import Queue
from threading import Lock, Thread
from time import sleep
import numpy as np

from module.geoparser import CropImage, SourceImage

from .config import CONFIG
from .task_result import TaskResult

@dataclass
class QueueElement():
    crop_data: np.ndarray
    crop_name: str
    task_id: int

def _process_loop(queue: list[QueueElement], lock: Lock, target_dict: dict[int,TaskResult], layouts: list[SourceImage]):
    while(True):
        if(len(queue)):
            start_time = date.today().strftime('%Y-%m-%dT%h:%m:%s')
            lock.acquire(True)
            data: QueueElement = queue.pop()
            lock.release()
            result = TaskResult()
            result.start = start_time
            result.task_id = data.task_id

            crop = CropImage(data.crop_data, data.crop_name)
            result.ul, result.ur, result.br, result.bl = crop.get_coords(layouts[0]).tolist()
            result.crs = layouts[0].epsg
            result.crop_name = data.crop_name
            result.layout_name = layouts[0].name
            result.end = date.today().strftime('%Y-%m-%dT%h:%m:%s')

            lock.acquire(True)
            target_dict[result.task_id] = result
            lock.release()
            print('processed')
            print(result.get_json())
        else:
            sleep(0.01)

class ProcessQueue():
    queue: list[QueueElement]
    lock: Lock
    target_dict: dict[int,TaskResult]
    layouts: SourceImage
    thread: Thread

    def __init__(self, lock: Lock, target_dict: dict[int,TaskResult], layouts: list[SourceImage]) -> None:
        self.queue = []
        self.lock = lock
        self.target_dict = target_dict
        self.thread = Thread(target=_process_loop, args=[self.queue, self.lock, self.target_dict, layouts])
        self.thread.start()
        
    def add_task(self, crop_data: np.ndarray, crop_name: str, task_id: int):
        self.lock.acquire(True)
        self.queue.append(QueueElement(crop_data, crop_name, task_id))
        self.lock.release()