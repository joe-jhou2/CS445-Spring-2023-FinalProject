import numpy as np


def estimateLocalTransforms_givenOutput(obj, invBk_perPatch_44P, G_34, epsilon, gamma):
    P = obj.linearIndicesInWindow_NP.shape[1]  # number of patches

    # Gather matrices vk_bar(I), vk_bar(M), vk(T), vk(O), for each patch
    Iban_4NP = obj.imgInput_4R[:, obj.linearIndicesInWindow_NP].reshape(4, -1)
    Mar_4NP = obj.imgMatch_4R[:, obj.linearIndicesInWindow_NP].reshape(4, -1)
    T_3NP = obj.imgTarget_3R[:, obj.linearIndicesInWindow_NP].reshape(3, -1)
    O_3NP = obj.imgOutput_3R[:, obj.linearIndicesInWindow_NP].reshape(3, -1)

    # Compute inverse matrix Bk at each patch
    Bk_44P = np.linalg.inv(invBk_perPatch_44P)

    # We obtain Ak at each patch by vectorizing:
    # Ak_34 = (O*Ibar' + epsilon*T*Mbar' + gamma*G) * Bk;
    O_Iban_t_34P = np.matmul(O_3NP, Iban_4NP.T)
    T_Mar_t_34P = np.matmul(T_3NP, Mar_4NP.T)
    leftSide_34P = (O_Iban_t_34P + epsilon*T_Mar_t_34P + gamma*G_34) @ Bk_44P
    Ak_perPatch_34P = leftSide_34P.reshape(3, 4, P).transpose(2, 0, 1)

    return Ak_perPatch_34P

