import cv2
import numpy as np
import shapely
import geotiff
from osgeo import gdal
import PIL.Image

SIFT = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


class _Image:
    image: np.ndarray
    descriptors: any
    keypoints: any

    def __init__(self) -> None:
        pass

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
        self.__read_tokens(image_path)
        self.image_file = geotiff.GeoTiff(image_path, crs_code=32637)
        self.process_image()

    def to_numpy(self):
        zarr_array = self.image_file.read()
        return np.array(zarr_array)

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
        self.image = np.array(self.to_numpy()[:, :, :3] / 1000 * 255, dtype=np.uint8)
        self.image = np.array(
            PIL.Image.fromarray(self.image).resize(
                [self.image.shape[0] // 4, self.image.shape[1] // 4],
                resample=PIL.Image.BICUBIC,
            )
        )
        return super().process_image()


class CropImage(_Image):
    def __init__(self, image: np.ndarray):
        super().__init__()

        self.image = image
        self.process_image()

    def match(self, source: SourceImage):
        matches = bf.match(self.descriptors, source.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > 50:
            src_pts = np.float32(
                [self.keypoints[m.queryIdx].pt for m in matches[:50]]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [source.keypoints[m.trainIdx].pt for m in matches[:50]]
            ).reshape(-1, 1, 2)
        transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return transform, mask

    def get_coords(self, source: SourceImage):
        M, _ = self.match(source)
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
