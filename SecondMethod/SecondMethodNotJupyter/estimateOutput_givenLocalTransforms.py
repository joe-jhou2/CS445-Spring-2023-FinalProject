import numpy as np



def estimateOutput_givenLocalTransforms(obj, Ak_perPatch_34P):
    P = obj.linearIndicesInWindow_NP.shape[1]  # number of patches
    R = obj.imgInput_3R.shape[1]  # total number of pixels

    # Gather matrix vk_bar(I) for each patch
    Iban_4NP = np.reshape(obj.imgInput_4R[:, obj.linearIndicesInWindow_NP.ravel(order='F')],
                          (4, obj.linearIndicesInWindow_NP.shape[1]), order='F')

    # Each patch results in (soft) constraints on several pixels
    outputConstraints_3NP = np.matmul(Ak_perPatch_34P, Iban_4NP)

    # Initialize arrays that contain the diagonal of the left-side matrix,
    # and the right-side.
    # TODO: turn this into an accumarray
    digM_R = np.zeros(R)
    u_3R = np.zeros((3, R))
    # ignore patches where local transform is NaN
    patchesNotNaN = np.where(np.sum(np.isnan(Ak_perPatch_34P.reshape((-1, P))), axis=0) == 0)[0]
    for k in patchesNotNaN:
        linearIndicesInWindow = obj.linearIndicesInWindow_NP[:, k]
        digM_R[linearIndicesInWindow] += 1
        u_3R[:, linearIndicesInWindow] += outputConstraints_3NP[:, :, k]

    obj.imgOutput_3R = np.divide(u_3R, digM_R, out=np.zeros_like(u_3R), where=(digM_R != 0))
