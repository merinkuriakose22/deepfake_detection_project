from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import cv2

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB):
    m = mse(imageA, imageB)
    s = compare_ssim(imageA, imageB, channel_axis=2)  # Specify channel axis for color images
    return m, s

