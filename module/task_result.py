import json


class TaskResult:
    task_id: int
    layout_name: str
    crop_name: str
    crs: str
    start: str
    end: str
    ul: list[float]
    ur: list[float]
    br: list[float]
    bl: list[float]

    def get_json(self):
        return json.dumps(self.__dict__, )