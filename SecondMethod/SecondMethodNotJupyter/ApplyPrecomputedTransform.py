import numpy as np
import LocalTransfer_affineModel


def ApplyPrecomputedTransform(imgInput, localTransform_array3CR, patchWidth):
    if patchWidth == 1:
        imgOutput = fastApplyTransforms(imgInput, localTransform_array3CR)
    else:
        localTransferRec = LocalTransfer_affineModel(imgInput, None, None)
        localTransferRec.gatherSquarePatches(patchWidth)

        # If cached transform is diagonal or linear, extend it to affine
        if localTransform_array3CR.shape[1] < 4:
            localTransform_array3CR = np.pad(localTransform_array3CR, ((0, 0), (0, 4 - localTransform_array3CR.shape[1]), (0, 0)), 'constant')

        # Get the linear indices of the patch centers
        patchCenters_linearIndex_1P = localTransferRec.linearIndicesInWindow_NP(np.ceil((patchWidth ** 2) / 2).astype(int))

        # Find the per-patch transform using the patch centers
        Ak_perPatch_34P = localTransform_array3CR[:, :, patchCenters_linearIndex_1P]
        # assert(np.isnan(Ak_perPatch_34P).sum() == 0)

        # Find the per-patch transform using the mask
        # Ak_perPatch_34P = localTransform_array3CR[:, :, imgMask_HW > 0]

        # Apply the transforms on the input image
        localTransferRec.estimateOutput_givenLocalTransforms(Ak_perPatch_34P)

        imgOutput = np.reshape(np.transpose(localTransferRec.imgOutput_3R), localTransferRec.imgSize)

    return imgOutput


def fastApplyTransforms(imgInput, localTransform_array3CR):
    C = localTransform_array3CR.shape[1]
    R = localTransform_array3CR.shape[2]
    assert (imgInput.shape[0] * imgInput.shape[1]) == R

    imgInput_31R = np.reshape(np.transpose(np.reshape(imgInput, (R, 3))), (3, 1, R))
    if C == 4:
        imgInput_C1R = np.vstack((imgInput_31R, np.ones((1, 1, R))))
    else:
        imgInput_C1R = imgInput_31R

    imgOutput_31R = localTransform_array3CR @ imgInput_C1R
    imgOutput = np.transpose(np.reshape(np.transpose(imgOutput_31R), (3, R)), (1, 0)).reshape(imgInput.shape)

    return imgOutput

