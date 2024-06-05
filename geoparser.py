import numpy as np
import shapely
import geotiff
from osgeo import gdal


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
        # self.image = self.to_numpy(self.__image_file)
        # print(self.image.shape)

    def to_numpy(self, geotiff_input: geotiff.GeoTiff):
        zarr_array = geotiff_input.read()
        return np.array(zarr_array)

    def __read_tokens(self, image_path: str):
        """Will get coords of the file in the source dataset. p1. """
        tiff = geotiff.GeoTiff(image_path)
        bounding_box = tiff.tif_bBox

        # Извлекаем координаты углов bounding box
        min_x, min_y = bounding_box[0]
        max_x, max_y = bounding_box[1]

        # Создаем список координат для полигона (обход по часовой стрелке)
        coords = [
            (min_x, min_y),  # нижний левый угол
            (max_x, min_y),  # нижний правый угол
            (max_x, max_y),  # верхний правый угол
            (min_x, max_y),  # верхний левый угол
        ]
        # Создаем экземпляр shapely.Polygon
        self.coords = shapely.Polygon(coords)
        self.epsg = str(tiff.crs_code)


class CropImage(_Image):
    pass

# test = SourceImage("data/layouts/layout_2021-06-15.tif")
# print(test.epsg)
