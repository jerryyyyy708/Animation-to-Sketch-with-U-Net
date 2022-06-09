import os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter


def sharpImage(img, sigma, k_sigma, p):
    sigma_large = sigma * k_sigma
    G_small = cv2.GaussianBlur(img, (0, 0), sigma)
    G_large = cv2.GaussianBlur(img, (0, 0), sigma_large)
    S = (1 + p) * G_small - p * G_large
    return S


def softThreshold(SI, epsilon, phi):
    T = np.zeros(SI.shape)
    SI_bright = (SI >= epsilon)
    SI_dark = (SI < epsilon)
    T[SI_bright] = 1.0
    T[SI_dark] = 1.0 + np.tanh(phi * (SI[SI_dark] - epsilon))
    return T


def xdog(img, sigma, k_sigma, p, epsilon, phi):
    S = sharpImage(img, sigma, k_sigma, p)
    SI = np.multiply(img, S)
    T = softThreshold(SI, epsilon, phi)
    return T

def sketch_process(filename):
    img = Image.open(filename)
    img = ImageOps.grayscale(img)
    img = img.filter(ImageFilter.FIND_EDGES)
    img = img.filter(ImageFilter.SMOOTH)
    img = ImageOps.invert(img)
    img.save("test.png")


def add_intensity(img, intensity=1.7):
    const = 255.0 ** (1.0 - intensity)
    img = (const * (img ** intensity))
    return img


def line_example_process(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img / 255.0
    sigma = np.random.choice([0.3, 0.4, 0.5])
    img = xdog(img, sigma, 4.5, 19, 0.01, 3)
    img = img * 255.0
    img = add_intensity(img)
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.tile(img, (1, 1, 3))
    return img


def get_sketch(src_path, dst_path):
    for i, origin_filename in enumerate(os.listdir(src_path)):
        print(origin_filename)
        try:
            origin_path = os.path.join(src_path, origin_filename)
            sketch_img = line_example_process(origin_path)

            sketch_path = os.path.join(dst_path, origin_filename)
            status = cv2.imwrite(sketch_path, sketch_img)
            if not status:
                raise Exception("Failed to write sketch file.")
        except Exception as e:
            print(f"Exception with file {origin_filename}.\nException message: {str(e)}.")


get_sketch("data_path", "sketch_path")
