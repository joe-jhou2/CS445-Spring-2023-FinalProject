import numpy as np


def perPixelLocalTransforms(obj, Ak_perPatch_3CP):
    N = obj.linearIndicesInWindow_NP.shape[0]  # number of pixels per patch
    R = obj.imgInput_3R.shape[1]  # number of pixels in image
    C = Ak_perPatch_3CP.shape[1]

    # Create C images, where each pixel shows some part of the local
    # transform estimated for the patch centered on this pixel
    localTransform_array3CR = np.empty((3, C, R))
    localTransform_array3CR[:] = np.nan
    patchCenterLinearIndex_1P = obj.linearIndicesInWindow_NP[N // 2, :]
    localTransform_array3CR[:, :, patchCenterLinearIndex_1P] = Ak_perPatch_3CP

    imgMask_HW = None
    if (nargout > 1):  # FIND OUT nargout!!!!
        isTransformNaN_1R = np.sum(np.isnan(
            localTransform_array3CR.reshape(3*C, R)), axis=0) > 0
        imgMask_HW = np.reshape(~isTransformNaN_1R, obj.imgSize)

    return localTransform_array3CR, imgMask_HW
