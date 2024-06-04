import numpy as np
import shapely
import geotiff

class _Image():
    image: np.ndarray

    def __init__(self) -> None:
        pass

    def process_image(self):
        """Will get image features. p2."""
        pass

class SourceImage(_Image):
    epsg: str
    coords: shapely.Polygon
    __image_file: geotiff.GeoTiff

    def __init__(self, image_path: str) -> None:
        self.__read_tokens(image_path)
        self.__image_file = geotiff.GeoTiff(image_path, crs_code=32637)

    def __read_tokens(self, image_path: str):
        """Will get coords of the file in the source dataset. p1. """
        pass

class CropImage(_Image):
    pass