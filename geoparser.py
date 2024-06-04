import numpy as np
import shapely
import geotiff

class GeoParser():
    epsg: str
    coords: shapely.Polygon
    image: np.ndarray

    def __init__(self, image_path: str) -> None:
        pass

    