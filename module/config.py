import os
import json

class Config():
    queue_max_size: int

    cache_layouts: bool

    max_matches_to_calc_proj: int
    max_affine_transform = 0.01
    homografy_confidence: int

    data_path: str
    layouts_dir_name: str
    crops_dir_name: str

    layouts_downscale: int
    layout_clip_max: int

    fix_search_window: int
    fix_repair_window: int
    fix_min_exp: float
    fix_max_exp: float

    crop_scale: float
    
    
    def __init__(self):
        self.cache_layouts = True
        self.queue_max_size = 100
        self.max_matches_to_calc_proj = 50
        self.homografy_confidence = 5.0

        self.data_path = 'data'
        self.layouts_dir_name = 'layouts'
        self.crops_dir_name = 'crops'

        self.layouts_downscale = 4
        self.layout_clip_max = 2000

        self.fix_search_window = 5
        self.fix_repair_window = 5
        self.fix_min_exp = 15
        self.fix_max_exp = 200

        self.crop_scale = 1

        if(os.path.exists('config.json')):
            with open('config.json', 'r') as  file:
                config = json.loads(file.read())

                self.__dict__.update(config)


    def __repr__(self) -> str:
        return str(self.__dict__)
    
CONFIG = Config()