import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from LocalTransfer_affineModel import LocalTransfer_affineModel


scaleFactor = 0.5
useModel = 'affine'

useClosedForm = 1
nbIterations = 3

# Load example input data and create synthetic match/target images
imgA = imread('imageA.jpg').astype(np.float64) / 255
output_shape = tuple((np.array(imgA.shape[:2]) * scaleFactor).astype(int))
imgA = resize(imgA, output_shape)

imgB = imread('imageB.jpg').astype(np.float64) / 255
output_shape = tuple((np.array(imgB.shape[:2]) * scaleFactor).astype(int))
imgB = resize(imgB, output_shape)

imgA = np.clip(imgA, 0, 1)
imgB = np.clip(imgB, 0, 1)

imgInput = imgA
imgMatch = imgA

imgTarget = imgB

# Our target image corresponds to the input, scaled in each color channel,
# with added intensity. We applied an affine transform in the left half,
# and a linear transform in the right half of the image.
imgTarget = imgA * np.reshape([0.9, 0.8, 0.7], [1, 1, 3])
imgTarget[:, :imgTarget.shape[1]//2, :] = -0.3 + imgTarget[:, :imgTarget.shape[1]//2, :] * np.reshape([1.5, 1.3, 1], [1, 1, 3])

# plt.imshow(imgTarget)
# plt.show()

# plt.imshow(imgInput)
# plt.show()


# localTransfer = LocalTransfer_affineModel(imgInput, imgMatch, imgTarget)