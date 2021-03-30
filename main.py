import cv2, glob
import numpy as np
import matplotlib.pyplot as plt
from tools.ImperfectionRecognizer import ImperfectionRecognizer

"""
This script performs a detection of the cracks and knots present in a piece of wood.
"""

if __name__ == '__main__':
    # Variables
    paths = glob.glob(f'./Samples/Test/*.png')
    ir = ImperfectionRecognizer(threshold_factor=2)

    # Load image
    for path in paths:
        img = cv2.imread(path)

        # Get imperfections mask and quantification
        mask, percent = ir.imperfections_and_quantification(img)

        # Show info
        print(f"Image: {path.split('/')[-1]}, percentage of imperfections: {percent}")

        fig, axs = plt.subplots(1,2)
        axs[0].imshow(mask)
        axs[0].set_title("Mask")
        axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Original")
        fig.suptitle(f'{path}')
        plt.show()

