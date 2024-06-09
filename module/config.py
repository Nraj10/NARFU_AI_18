import os
import json

class Config():
    max_matches_to_calc_proj: int
    homografy_confidence: int

    data_path: str
    layouts_dir_name: str
    crops_dir_name: str

    layouts_downscale: int
    layout_clip_max: int
    
    
    def __init__(self):
        self.max_matches_to_calc_proj = 50
        self.homografy_confidence = 5.0

        self.data_path = 'data'
        self.layouts_dir_name = 'layouts'
        self.crops_dir_name = 'crops'

        self.layouts_downscale = 4
        self.layout_clip_max = 2000

        if(os.path.exists('config.json')):
            with open('config.json', 'r') as  file:
                config = json.loads(file.read())

                self.__dict__.update(config)


    def __repr__(self) -> str:
        return str(self.__dict__)
    
CONFIG = Config()