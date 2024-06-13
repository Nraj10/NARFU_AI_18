import json
import pandas as pd

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
    fix_info: str

    def get_json(self):
        return json.dumps(self.__dict__, )
    
    def get_csv(self):
        pd.read_json(self.get_json).to_csv('coords.csv')