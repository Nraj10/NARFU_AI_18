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
        self.ul = str(self.ul)[1:-1].replace(',', ';')
        self.ur = str(self.ur)[1:-1].replace(',', ';')
        self.br = str(self.br)[1:-1].replace(',', ';')
        self.bl = str(self.bl)[1:-1].replace(',', ';')
        self.fix_info = self.fix_info.replace('\n', '\t')
        df = pd.read_json(f'[{self.get_json()}]')
        df.transpose().to_csv('coords.csv', sep='\t', header=False)