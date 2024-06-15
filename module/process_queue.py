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
class QueueElement:
    crop_data: np.ndarray
    crop_name: str
    task_id: int


def _process_task(data: QueueElement, layouts: list[SourceImage]):
    result = TaskResult()
    crop = CropImage(data.crop_data, data.crop_name)
    result.task_id = data.task_id
    result.crs = f"EPSG:{layouts[0].epsg}"
    result.crop_name = data.crop_name
    result.fix_info = crop.fix_info

    matrices = []
    for layout in layouts:
        try:
            matrices.append(crop.match(layout)[0])
        except Exception:
            pass

    products = np.array([abs(x.prod()) for x in matrices])
    closest_match_product = products.min()
    if (
        (len(matrices) == 0)
        or (closest_match_product > CONFIG.max_affine_transform)
    ):
        print(crop.name, closest_match_product)
        result.set_end_time()
        result.layout_name = ""
        return result

    index = np.where(products == closest_match_product)[0][0]
    layout = layouts[index]

    result.ul, result.ur, result.br, result.bl = crop.get_coords(
        layout, matrices[index]
    ).tolist()
    result.layout_name = layout.name
    result.set_end_time()
    return result


def _process_loop(
    queue: Queue,
    lock: Lock,
    target_dict: dict[int, TaskResult],
    layouts: list[SourceImage],
):
    while True:
        if not queue.empty():
            data: QueueElement = queue.get()
            if data == 0:
                break
            result = _process_task(data, layouts)
            lock.acquire(True)
            target_dict[result.task_id] = result
            lock.release()
            print("processed")
            print(result.get_json())
        else:
            sleep(0.01)


class ProcessQueue:
    queue: Queue
    lock: Lock
    target_dict: dict[int, TaskResult]
    layouts: SourceImage
    thread: Thread

    def __init__(
        self, lock: Lock, target_dict: dict[int, TaskResult], layouts: list[SourceImage]
    ) -> None:
        self.queue = Queue(CONFIG.queue_max_size)
        self.lock = lock
        self.target_dict = target_dict
        self.thread = Thread(
            target=_process_loop,
            args=[self.queue, self.lock, self.target_dict, layouts],
        )
        self.layouts = layouts
        self.thread.start()

    def add_task(self, crop_data: np.ndarray, crop_name: str, task_id: int):
        self.queue.put(QueueElement(crop_data, crop_name, task_id))

    def process_without_queue(self, crop_data: np.ndarray, crop_name: str):
        return _process_task(QueueElement(crop_data, crop_name, -1), self.layouts)

    def destroy(self):
        self.queue.put(0)
