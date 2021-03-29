import cv2
import numpy as np
import os

def read_yaml(root):
    """
    Read yaml file
    :param root: yaml root
    :return: np.darray
    """

    fs = cv2.FileStorage(root, cv2.FILE_STORAGE_READ)
    fn = fs.getNode("rectangles")

    return fn.mat()

def label2mask(rootYaml, image_shape):
    """
    Create a binary mask representing the imperfection label in the image.
    :param rootYaml: yaml path of the image defining the mask
    :param image_shape: shape of the image
    :return: mask
    """

    mask = np.zeros((image_shape[0], image_shape[1]))

    if(os.path.isfile(rootYaml)):
        rects = read_yaml(rootYaml)
        for i in range(0,len(rects[0])):
            y = int(rects[0][i])
            x = int(rects[1][i])
            y_end = y + int(rects[2][i])
            x_end = x + int(rects[3][i])
            mask[x:x_end,y:y_end] = 1

    return mask