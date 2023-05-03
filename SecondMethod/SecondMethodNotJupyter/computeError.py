import numpy as np
import math


def computeError(self):
    # IMPORTANT: This method assumes that the input and match images
    # are identical. It computes the error between the transformed input
    # (or transformed match) and the target image.

    # Compute the per-pixel squared error
    perPixelError_1R = np.sum((self.imgTarget_3R - self.imgOutput_3R) ** 2, axis=1)

    perPixelError_1R = np.sqrt(perPixelError_1R)

    # Sum the error over all pixels of every patch; store the error at the
    # center pixel.
    N = self.linearIndicesInWindow_NP.shape[0]
    P = self.linearIndicesInWindow_NP.shape[1]
    centerPixel_linearIndex_1P = self.linearIndicesInWindow_NP[math.ceil(N / 2) - 1, :]
    perPixelError_NP = np.reshape(perPixelError_1R[self.linearIndicesInWindow_NP], (N, P))
    perPatchError_1R = np.full_like(perPixelError_1R, np.nan)
    perPatchError_1R[centerPixel_linearIndex_1P] = np.sum(perPixelError_NP, axis=0)

    return perPixelError_1R, perPatchError_1R
