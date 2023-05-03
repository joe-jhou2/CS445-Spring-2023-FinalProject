import numpy as np


def gatherSquarePatches(obj, patchWidth):
    halfK = patchWidth // 2
    imageWidth = obj.imgSize[1]
    imageHeight = obj.imgSize[0]

    # Get the coordinate of the center pixel of all patches in the images,
    # except near the borders
    centerPixel_x = np.tile(np.arange(1 + halfK, imageWidth - halfK), (imageHeight - 2 * halfK, 1))
    centerPixel_y = np.tile(np.arange(1 + halfK, imageHeight - halfK)[:, np.newaxis], (1, imageWidth - 2 * halfK))
    centerPixel_linearIndices = np.ravel_multi_index((centerPixel_y, centerPixel_x), (imageHeight, imageWidth))

    # Get the linear index offsets of all neighbors in a window
    neighbors_offset_col = np.tile(np.arange(-halfK, halfK + 1), (patchWidth, 1))
    neighbors_offset_row = neighbors_offset_col.T
    neighbors_offset_linearIndices = neighbors_offset_col * imageHeight + neighbors_offset_row

    # Get the linear index of each neighbor in each patch
    N = neighbors_offset_linearIndices.size
    patchesLinearIndices_NP = centerPixel_linearIndices.reshape(-1, 1) + neighbors_offset_linearIndices.ravel()

    # Discard patches that contain at least one NaN pixel
    badPatchIndices = findBadPatchIndices(obj.imgInput_3R, patchesLinearIndices_NP)
    if obj.imgMatch_3R is not None:
        badPatchIndices = np.concatenate((badPatchIndices, findBadPatchIndices(obj.imgMatch_3R, patchesLinearIndices_NP)))
    if obj.imgTarget_3R is not None:
        badPatchIndices = np.concatenate((badPatchIndices, findBadPatchIndices(obj.imgTarget_3R, patchesLinearIndices_NP)))
    patchesLinearIndices_NP = np.delete(patchesLinearIndices_NP, badPatchIndices, axis=1)

    obj.linearIndicesInWindow_NP = patchesLinearIndices_NP
    obj.sparseLinearIndices_pattern_row = np.tile(np.arange(N), (N, 1))  # used for computing neighborhoods in the adjacency matrices later


def findBadPatchIndices(img_3R, patchesLinearIndices_NP):
    N, P = patchesLinearIndices_NP.shape

    pixelValues_3NP = img_3R[:, patchesLinearIndices_NP.reshape(-1)]

    nbNaNValuesPerPatch_1P = np.sum(np.isnan(pixelValues_3NP.reshape(3, N, P)), axis=0)

    return np.nonzero(nbNaNValuesPerPatch_1P > 0)[0]
