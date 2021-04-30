import cv2
import argparse;
from tools.ImperfectionRecognizer import ImperfectionRecognizer

"""
This script performs a detection of the cracks and knots present in a piece of wood.
"""

def calculate_imperfections(path_image):

    # Variables
    ir = ImperfectionRecognizer(threshold_factor=2)

    # Load image
    img = cv2.imread(path_image)

    # Get imperfections mask and quantification
    mask, percent = ir.imperfections_and_quantification(img)

    # Show info
    print(f"Image: {path_image.split('/')[-1]}, percentage of imperfections: {percent}")


#python calculate_imperfections.py -image=./Samples/Train/1
ap = argparse.ArgumentParser()
ap.add_argument("-image", required=False, help="path to where image resides")
args = vars(ap.parse_args())
path_image = args['image']

# Sino me pasan argumentos tomo una ruta predefinida
if path_image == None:
    path_image = './Samples/Train/1'
path_image += '.png'
calculate_imperfections(path_image)