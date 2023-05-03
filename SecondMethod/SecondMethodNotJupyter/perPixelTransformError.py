import numpy as np


def perPixelTransformError(obj, localTransform_array3CR):
    N = obj.linearIndicesInWindow_NP.shape[0]  # number of pixels per patch
    P = obj.linearIndicesInWindow_NP.shape[1]  # number of patches
    R = obj.imgInput_3R.shape[1]  # total number of pixels
    C = localTransform_array3CR.shape[1]

    # Gather matrix vk(M) and vk(T) for each patch
    if C == 3:
        M_CNP = np.reshape(obj.imgMatch_3R[:, obj.linearIndicesInWindow_NP.flatten()],
                           (3, *obj.linearIndicesInWindow_NP.shape))
    elif C == 4:
        M_CNP = np.reshape(obj.imgMatch_4R[:, obj.linearIndicesInWindow_NP.flatten()],
                           (4, *obj.linearIndicesInWindow_NP.shape))
    T_3NP = np.reshape(obj.imgTarget_3R[:, obj.linearIndicesInWindow_NP.flatten()],
                       (3, *obj.linearIndicesInWindow_NP.shape))

    # Get linear index of the center of each pixel
    centerPixel_linearIndex_1P = obj.linearIndicesInWindow_NP[np.ceil(N / 2).astype(int) - 1, :]
    Ak_perPatch_3CP = localTransform_array3CR[:, :, centerPixel_linearIndex_1P]

    # Apply the local transforms on each neighborhood
    transformedM_3NP = np.matmul(Ak_perPatch_3CP, M_CNP)

    # Sum the error over all pixels of every patch; store the error at the
    # center pixel.
    perTransformError_1P = np.sum(np.reshape((T_3NP - transformedM_3NP) ** 2, (-1, P)), axis=0)
    perTransformError_1R = np.full((1, R), np.nan)
    perTransformError_1R[:, centerPixel_linearIndex_1P] = perTransformError_1P

    return perTransformError_1R
