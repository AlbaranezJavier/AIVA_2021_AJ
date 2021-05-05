import cv2, glob
import matplotlib.pyplot as plt
from tools.ImperfectionRecognizer import ImperfectionRecognizer
from tools.LabelsManager import label2mask

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
        path_gt = '.' + path.split('.')[1] + '.reg'

        # Get imperfections mask and quantification
        mask, percent = ir.imperfections_and_quantification(img)

        mask_gt = label2mask(path_gt, img.shape)
        # Show info
        print(f"Image: {path.split('/')[-1]}, percentage of imperfections: {percent}")

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original")
        axs[1].imshow(mask_gt)
        axs[1].set_title("GT")
        axs[2].imshow(mask)
        axs[2].set_title("Mask")
        fig.suptitle(f'{path}')
        plt.show()

