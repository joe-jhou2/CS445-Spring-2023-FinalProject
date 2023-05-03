import numpy as np
from scipy.sparse import triu


def transfer_closedForm(obj):
    # Parameters
    epsilon = 1
    gamma = 0.01

    # Compute global transform, and globally transformed image
    globalTransform_3C = obj.computeGlobalTransform()

    # Construct system matrix and right side
    M_sparseRR, u_3R, invBk_perPatch_CCP = obj.buildClosedFormMatrices(globalTransform_3C, epsilon, gamma)

    # Make sure matrix M is symmetric
    # (sometimes there are some inaccuracies, and M-M'=1e-14 in some cells)
    # by replacing the lower triangle by the transpose of the upper triangle
    M_sparseRR = triu(M_sparseRR) + triu(M_sparseRR, 1).transpose()

    # Solve linear system
    obj.imgOutput_3R = np.linalg.solve(M_sparseRR, u_3R)

    # Output transformed image
    imgOutput = obj.imgOutput_3R.transpose().reshape(obj.imgSize)

    # Estimate the best-matching local transforms
    Ak_perPatch_3CP = obj.estimateLocalTransforms_givenOutput(invBk_perPatch_CCP, globalTransform_3C, epsilon, gamma)

    # Output the per-pixel local transform and corresponding binary mask
    localTransform_array3CR, imgMask_HW = obj.perPixelLocalTransforms(Ak_perPatch_3CP)

    return imgOutput, localTransform_array3CR, imgMask_HW
