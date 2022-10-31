import cv2
from cv2 import cvtColor
from cv2 import COLOR_GRAY2BGR
import numpy as np
from time import time


def add_salt_and_pepper(image, low_th, high_th):
    saltpepper_noise = np.zeros_like(image)
    cv2.randu(saltpepper_noise, 0, 255)

    image[saltpepper_noise < low_th] = 0
    image[saltpepper_noise > high_th] = 255


class Rectangle:
    def __init__(self, x_tl, y_tl, width, height):
        self.x_tl = x_tl
        self.y_tl = y_tl
        self.width = width
        self.height = height

    def intersect(self, rect):
        x_tl = max(self.x_tl, rect.x_tl)
        y_tl = max(self.y_tl, rect.y_tl)
        x_br = min(self.x_tl + self.width, rect.x_tl + rect.width)
        y_br = min(self.y_tl + self.height, rect.y_tl + rect.height)
        if x_tl < x_br and y_tl < y_br:
            return Rectangle(x_tl, y_tl, x_br - x_tl, y_br - y_tl)
        return None

    __and__ = intersect

    def equal(self, rect):
        x_tl_diff = self.x_tl - rect.x_tl
        y_tl_diff = self.y_tl - rect.y_tl
        x_br_diff = self.width - rect.width
        y_br_diff = self.height - rect.height
        diff = x_tl_diff + y_tl_diff + x_br_diff + y_br_diff
        if diff == 0:
            return True
        return False

    __eq__ = equal


class EROS:

    def __init__(self, kernel_size, frame_height, frame_width, decay_base=0.3):
        self.kernel_size = kernel_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.decay_base = decay_base
        self._image = np.zeros((frame_height, frame_width), dtype=np.uint8)

        if self.kernel_size % 2 != 0:
            self.kernel_size += 1

    def get_frame(self):
        return self._image

    def update(self, vx, vy):
        odecay = self.decay_base ** (1.0 / self.kernel_size)
        half_kernel = int(self.kernel_size / 2)
        roi_full = Rectangle(0, 0, self.frame_width, self.frame_height)
        roi_raw = Rectangle(0, 0, self.kernel_size, self.kernel_size)

        roi_raw.x_tl = vx - half_kernel
        roi_raw.y_tl = vy - half_kernel
        roi_valid = roi_raw & roi_full

        if roi_valid is None:
            return True
        roi = [roi_valid.y_tl, roi_valid.y_tl + roi_valid.height, roi_valid.x_tl, roi_valid.x_tl + roi_valid.width]
        update_mask = np.ones((roi[1] - roi[0], roi[3] - roi[2]), dtype=np.float) * odecay
        self._image[roi[0]:roi[1], roi[2]:roi[3]] = np.multiply(self._image[roi[0]:roi[1], roi[2]:roi[3]],
                                                                update_mask).astype(np.uint8)
        self._image[vy, vx] = 255

        return roi_raw != roi_valid


class TOS:

    def __init__(self, kernel_size, line_thickness, frame_height, frame_width):
        self.kernel_size = kernel_size
        self.line_thickness = line_thickness
        self.frame_height = frame_height
        self.frame_width = frame_width
        self._image = np.zeros((frame_height, frame_width), dtype=np.uint8)

    def get_frame(self):
        return self._image

    def update(self, vx, vy):
        thick_thresh = 255 - self.kernel_size * self.line_thickness
        half_kernel = int(self.kernel_size / 2)
        roi_full = Rectangle(0, 0, self.frame_width, self.frame_height)
        roi_raw = Rectangle(0, 0, self.kernel_size, self.kernel_size)

        roi_raw.x = vx - half_kernel
        roi_raw.y = vy - half_kernel
        roi_valid = roi_raw & roi_full

        if roi_valid is None:
            return True

        roi = [roi_valid.y_tl, roi_valid.y_tl + roi_valid.height, roi_valid.x_tl, roi_valid.x_tl + roi_valid.width]
        roi_img = self._image[roi[0]:roi[1], roi[2]:roi[3]]
        update_indices = roi_img < thick_thresh
        roi_img[update_indices] = 0
        update_indices = roi_img >= thick_thresh
        roi_img[update_indices] -= 1

        self._image[vy, vx] = 255

        return roi_raw != roi_valid


class EROSSynthetic:

    def get_frame(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, self.gaussian_blur_k, self.gaussian_blur_sigma)
        image = cv2.Canny(image, threshold1=self.canny_low_th, threshold2=self.canny_high_th,
                          apertureSize=self.canny_aperture, L2gradient=self.canny_l2_grad)

        # add pepper only: this adds noise to the canny edges, making them
        # look like closer to the real eros
        add_salt_and_pepper(image, 90, 255)

        add_salt_and_pepper(image, self.salt_pepper_low_th, self.salt_pepper_high_th)
        image = cv2.GaussianBlur(image, self.gaussian_blur_k, self.gaussian_blur_sigma)

        return image

    def __init__(self, gaussian_blur_k_size=5, gaussian_blur_sigma=0, canny_low_th=0, canny_high_th=1000,
                 canny_aperture=5, canny_l2_grad=False, salt_pepper_low_th=30, salt_pepper_high_th=225):
        # gaussian blur params
        self.gaussian_blur_k = (gaussian_blur_k_size, gaussian_blur_k_size)
        self.gaussian_blur_sigma = gaussian_blur_sigma

        # canny edge detection params
        self.canny_low_th = canny_low_th
        self.canny_high_th = canny_high_th
        self.canny_aperture = canny_aperture
        self.canny_l2_grad = canny_l2_grad

        # salt and pepper params
        self.salt_pepper_low_th = salt_pepper_low_th
        self.salt_pepper_high_th = salt_pepper_high_th


class PIM:

    def __init__(self, frame_height, frame_width, tau=1.0):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.tau = tau
        self.ts = 0
        self._image = np.zeros((frame_height, frame_width), dtype=float)
        self._normed = np.zeros((frame_height, frame_width), dtype=float)
        self._normed8U = np.zeros((frame_height, frame_width), dtype=np.uint8)
        self._normed3ch = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    def get_frame(self):
        return self._image

    def get_normed_rgb(self):
        # get the minimum and maximum values
        min_val, max_val, _, _ = cv2.minMaxLoc(self._image)
        # find the absolute maximum
        max_val = max(abs(max_val), abs(min_val))
        if max_val > 0:
            self._normed = ((self._image / (2 * max_val)) + 0.5) * 255
        self._normed8U = np.uint8(self._normed)
        cv2.cvtColor(self._normed8U, cv2.COLOR_GRAY2BGR, self._normed3ch)
        return self._normed3ch

    def update(self, vx, vy, p):
        if p > 0:
            self._image[vy, vx] += 1.0
        else:
            self._image[vy, vx] -= 1.0

    def perform_decay(self, timestamp):
        self._image *= np.exp(self.tau * (self.ts - timestamp))
        self.ts = timestamp


class eventFrame:

    def __init__(self, frame_height, frame_width, n=7500):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.n = n
        self.ts = 0
        self._image = np.zeros((frame_height, frame_width), dtype=float)
        self._normed = np.zeros((frame_height, frame_width), dtype=float)
        self.threshold = 0.1 / 255

    def get_frame(self):
        return self._image
    def reset_frame(self):
        self._image = np.zeros((self.frame_height, self.frame_width), dtype=float)

    def get_normed(self):
        # Calculate the mean of the image
        count_unique = np.count_nonzero(self._image)
        if count_unique == 0:
            count_unique = 1
        mean_val = (np.sum(np.sum(self._image))) / count_unique

        # calculate the variance of the (non-zero) pixels
        difference = self._image[self._image > 0] - mean_val
        var = (np.sum(np.square(difference))) / count_unique
        sigma = np.sqrt(var)
        if sigma < self.threshold:
            sigma = self.threshold

        # scale each pixel by the scale factor
        scale_factor = 255.0 / (3.0 * sigma)
        self._normed = self._image * scale_factor
        self._normed[self._normed > 255.0] = 255.0
        self._normed[self._normed < 0.0] = 0.0
        return self._normed

    def update(self, vx, vy):
        self._image[vy, vx] += 1.0
