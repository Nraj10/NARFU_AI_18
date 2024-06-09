import cv2
import numpy as np
import shapely
import geotiff
from osgeo import gdal
import PIL.Image
from tifffile import imread as tifread

from NARFU_AI_18.config import CONFIG

SIFT = cv2.SIFT_create(nfeatures=100000, enable_precise_upscale = True, sigma=1.0)



class _Image:
    image: np.ndarray
    descriptors: any
    keypoints: any
    name: str

    def __init__(self) -> None:
        self.name = ''
        pass

    def correct_gamma(self):
        invGamma = 1.0 / 1.5
        table = np.array([((i / 255.0) ** invGamma) * 255
		    for i in np.arange(0, 256)]).astype("uint8")
        self.image = cv2.LUT(self.image, table)

    def process_image(self):
        """Will get image features. p2."""
        self.keypoints, self.descriptors = SIFT.detectAndCompute(self.image, None)
        pass

    def draw_keypoints(self):
        return cv2.drawKeypoints(self.image, self.keypoints, self.image)


class SourceImage(_Image):
    epsg: str
    coords: shapely.Polygon
    image_file: geotiff.GeoTiff

    def __init__(self, image_path: str) -> None:
        super().__init__()
        self.name = image_path
        self.__read_tokens(image_path)
        self.image_file = geotiff.GeoTiff(image_path, crs_code=32637)
        self.process_image()

    def to_numpy(self):
        return tifread(self.name)

    def __read_tokens(self, image_path: str):
        """Will get coords of the file in the source dataset. p1."""
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

    def process_image(self):
        # TODO: add caching to pickle
        self.image = np.array(self.to_numpy()[:, :, :3] / CONFIG.layout_clip_max * 255)
        self.image = np.array(
            PIL.Image.fromarray(np.array(self.image, dtype=np.uint8)).resize(
                [self.image.shape[0] // 4, self.image.shape[1] // 4],
                resample=PIL.Image.BICUBIC,
            )
        )
        cv2.normalize(self.image, self.image, 0, 240, cv2.NORM_MINMAX)
        # self.correct_gamma()
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return super().process_image()


class CropImage(_Image):
    def __init__(self, image: np.ndarray, name = ''):
        super().__init__()

        self.image = image
        cv2.normalize(self.image, self.image, 0, 240, cv2.NORM_MINMAX)
        # self.image = image[:,:,3]
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.process_image()

    def match(self, source: SourceImage):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(self.descriptors, source.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > CONFIG.max_matches_to_calc_proj:
            src_pts = np.float32(
                [self.keypoints[m.queryIdx].pt for m in matches[:CONFIG.max_matches_to_calc_proj]]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [source.keypoints[m.trainIdx].pt for m in matches[:CONFIG.max_matches_to_calc_proj]]
            ).reshape(-1, 1, 2) 
        else:
            src_pts = np.float32(
                [self.keypoints[m.queryIdx].pt for m in matches[:]]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [source.keypoints[m.trainIdx].pt for m in matches[:]]
            ).reshape(-1, 1, 2) 
        transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, CONFIG.homografy_confidence)
        return transform, mask, matches

    def get_coords(self, source: SourceImage):
        M = self.match(source)[0]
        left_top_corner = M.dot(np.array([0, 0, 1]))
        right_top_corner = M.dot(np.array([self.image.shape[0], 0, 1]))
        right_bottom_corner = M.dot(
            np.array([self.image.shape[0], self.image.shape[1], 1])
        )
        left_bottom_corner = M.dot(np.array([0, self.image.shape[1], 1]))
        return np.array(
            [left_top_corner, right_top_corner, right_bottom_corner, left_bottom_corner]
        )


# test = SourceImage("data/layouts/layout_2021-06-15.tif")
# print(test.epsg)
