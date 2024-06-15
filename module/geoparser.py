import os
import pickle
import PIL.ImageOps
import cv2
import numpy as np
import shapely
import geotiff
from osgeo import gdal
import PIL.Image
from tifffile import imread as tifread
from scipy import ndimage
from module.config import CONFIG
import copyreg

SIFT = cv2.SIFT_create()  # nfeatures=100000, enable_precise_upscale = True, sigma=1.0

def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)


copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

class _Image:
    image: np.ndarray
    descriptors: any
    keypoints: any
    name: str

    def __init__(self) -> None:
        self.name = ""
        pass

    def correct_gamma(self):
        invGamma = 1.0 / 2.0
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        self.image = cv2.LUT(self.image, table)

    def equalize_channels(self):
        mean = 2*np.median(self.image)
        # + 1 * np.std(self.image, dtype=np.float32)
        self.image = self.image / mean
        print(mean, self.image.max())
        self.image *= 255
        self.image = np.array(self.image.clip(0, 255), dtype=np.uint8)

    def process_image(self):
        """Will get image features. p2."""
        self.keypoints, self.descriptors = SIFT.detectAndCompute(self.image, None)
        pass

    def draw_keypoints(self):
        return cv2.drawKeypoints(self.image, self.keypoints, self.image)

    def correct_broken_channels(self):
        win_rows = CONFIG.fix_search_window
        win_cols = CONFIG.fix_search_window
        median_window = CONFIG.fix_repair_window
        amin = CONFIG.fix_min_exp
        amax = CONFIG.fix_max_exp
        mwin = median_window // 2

        broken_channels = []
        broken_pixels = []
        fixed_pixels = []
        for i in range(4):
            channel = self.image[:, :, i]
            pixels = np.argwhere(
                (
                    ndimage.uniform_filter(channel, (win_rows, win_cols))
                    / 10
                    * amin
                    / 10
                    > channel
                )
                | (
                    ndimage.uniform_filter(channel, (win_rows, win_cols)) / 100 * amax
                    < channel
                )
            )
            broken_channels.append(pixels)
            channel_broken_pixels = []
            channel_fixed_pixels = []
            for pixel in pixels:
                channel_broken_pixels.append(channel[pixel[0], pixel[1]])
                val = np.median(
                    channel[
                        pixel[0] - mwin : pixel[0] + mwin,
                        pixel[1] - mwin : pixel[1] + mwin,
                    ]
                ).astype(int)
                channel_fixed_pixels.append(val)
                self.image[pixel[0], pixel[1], i] = val
            broken_pixels.append(channel_broken_pixels)
            fixed_pixels.append(channel_fixed_pixels)

        result = ""
        for ch_i in range(4):
            i = 0
            for pixel in broken_channels[ch_i]:
                result += f"{pixel[0]}; {pixel[1]}; {ch_i}; {int(broken_pixels[ch_i][i])}; {fixed_pixels[ch_i][i]}\n"
                i += 1
        return result


class SourceImage(_Image):
    epsg: str
    coords: shapely.Polygon
    width_m: float
    height_m: float
    _min_x: float
    _min_y: float
    image_file: geotiff.GeoTiff
    image_shape: list[int]

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
        self.coords = [
            (min_x, min_y),  # нижний левый угол
            (max_x, min_y),  # нижний правый угол
            (max_x, max_y),  # верхний правый угол
            (min_x, max_y),  # верхний левый угол
        ]
        print({"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y})
        self.width_m = max_x - min_x
        self.height_m = max_y - min_y
        self._min_x = min_x
        self._min_y = min_y
        # Создаем экземпляр shapely.Polygon
        # self.coords = shapely.Polygon(coords)
        self.epsg = str(tiff.crs_code)

    def get_pixel_coord(self, coord: list[float]):
        return (
            self._min_x + self.width_m * coord[0] / self.image_shape[0],
            self._min_y + self.height_m * coord[1] / self.image_shape[1],
        )

    def process_image(self):
        if os.path.exists(self.name + ".pkl") and CONFIG.cache_layouts:
            try:
                with open(self.name + ".pkl", "rb") as file:
                    archive = pickle.load(file)
                    self.__dict__.update(archive)
                    file.close()
                    return
            except Exception as e:
                print(e)

        self.image = np.array(self.to_numpy()[:, :, :3], dtype=np.float32).clip(0, 6000)
        self.equalize_channels()
        # mean = np.mean(self.image, dtype=np.float32)
        # # + 4 * np.std(
        # #     self.image, dtype=np.float32
        # # )
        # self.image = self.image / mean
        # print(mean, self.image.max())
        # self.image *= 255
        # self.image.clip(0, 255)
        self.image = PIL.Image.fromarray(self.image.astype(np.uint8)).resize(
            [
                int(self.image.shape[0] / CONFIG.layouts_downscale),
                int(self.image.shape[1] / CONFIG.layouts_downscale),
            ],
            resample=PIL.Image.BICUBIC,
        )
        # self.image = PIL.ImageOps.posterize(self.image, 3)
        self.image = PIL.ImageOps.grayscale(self.image)
        self.image = np.array(self.image)
        # self.image = self.image[:, :, :3]
        # cv2.normalize(self.image, self.image, 0, 240, cv2.NORM_MINMAX)
        # self.correct_gamma()
        self.image_shape = self.image.shape
        # if(self.image.shape[2] > 1):
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # self.image = cv2.bilateralFilter(self.image,3,50,15)
        # self.image = cv2.Canny(self.image, 20, 100, L2gradient=True)

        super().process_image()

        try:
            with open(self.name + ".pkl", 'wb+') as file:
                pickle.dump({
                    'coords': self.coords,
                    'image': self.image,
                    'image_shape': self.image_shape,
                    'descriptors': self.descriptors,
                    'keypoints': self.keypoints,
                }, file)
        except Exception as e:
            print(e)


class CropImage(_Image):
    fix_info: str

    def __init__(self, image: np.ndarray, name=""):
        super().__init__()

        self.image = image
        self.fix_info = self.correct_broken_channels()
        self.image = self.image[:, :, :3]
        self.name = name
        self.equalize_channels()
        # cv2.normalize(self.image, self.image, 0, 240, cv2.NORM_MINMAX)
        # self.image = image[:,:,3]
        # self.correct_gamma()
        self.image = PIL.Image.fromarray(self.image)
        self.image = PIL.ImageOps.scale(self.image, 0.5)
        # self.image = PIL.ImageOps.posterize(self.image, 3)
        self.image = PIL.ImageOps.grayscale(self.image)
        self.image = np.array(self.image)
        # if(self.image.shape[2] > 1):
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # self.image = cv2.GaussianBlur(self.image,(7,7),0)
        # self.image = cv2.bilateralFilter(self.image,3,50,15)
        # self.image = cv2.Canny(self.image, 20, 100, L2gradient=True)
        self.process_image()

    def match(self, source: SourceImage):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(self.descriptors, source.descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) > CONFIG.max_matches_to_calc_proj:
            src_pts = np.float32(
                [
                    self.keypoints[m.queryIdx].pt
                    for m in matches[: CONFIG.max_matches_to_calc_proj]
                ]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [
                    source.keypoints[m.trainIdx].pt
                    for m in matches[: CONFIG.max_matches_to_calc_proj]
                ]
            ).reshape(-1, 1, 2)
        else:
            src_pts = np.float32(
                [self.keypoints[m.queryIdx].pt for m in matches[:]]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [source.keypoints[m.trainIdx].pt for m in matches[:]]
            ).reshape(-1, 1, 2)
        transform, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, CONFIG.homografy_confidence
        )
        return transform, mask, matches

    def get_coords(self, source: SourceImage, M=None):
        if M is None:
            M = self.match(source)[0]
        luc = M.dot(np.array([0, 0, 1]))
        ruc = M.dot(np.array([self.image.shape[0], 0, 1]))
        rbc = M.dot(np.array([self.image.shape[0], self.image.shape[1], 1]))
        lbc = M.dot(np.array([0, self.image.shape[1], 1]))
        return np.array(
            [
                source.get_pixel_coord(luc),
                source.get_pixel_coord(ruc),
                source.get_pixel_coord(rbc),
                source.get_pixel_coord(lbc),
            ]
        )


# test = SourceImage("data/layouts/layout_2021-06-15.tif")
# print(test.epsg)
