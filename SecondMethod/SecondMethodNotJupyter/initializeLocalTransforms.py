import numpy as np
from numpy.linalg import inv


def initializeLocalTransforms(obj, G_34, epsilon, gamma):
    P = obj.linearIndicesInWindow_NP.shape[1]

    # Gather matrices vk_bar(M), vk(T), for each patch
    Mar_4NP = np.reshape(obj.imgMatch_4R[:, obj.linearIndicesInWindow_NP.flatten()],
                          [4, obj.linearIndicesInWindow_NP.shape[1]])
    T_3NP = np.reshape(obj.imgTarget_3R[:, obj.linearIndicesInWindow_NP.flatten()],
                       [3, obj.linearIndicesInWindow_NP.shape[1]])

    # We obtain Ak at each patch by vectorizing:
    # Ak_34 = (epsilon * (T*Mbar') + gamma*G) * inv(epsilon * (Mbar*Mbar') + gamma*eye(4));
    Mar_Mar_t_44P = np.matmul(Mar_4NP, Mar_4NP.transpose((1, 0))).reshape(4, 4, P)
    T_Mar_t_34P = np.matmul(T_3NP, Mar_4NP.transpose((1, 0))).reshape(3, 4, P)
    leftSide_34P = epsilon * T_Mar_t_34P.transpose((0, 2, 1)) + gamma * G_34[np.newaxis, ...]
    invRightSide_44P = epsilon * Mar_Mar_t_44P + gamma * np.eye(4)[np.newaxis, ...]

    # Inverse the right side for each patch
    rightSide_44P = np.zeros_like(invRightSide_44P)
    for k in range(P):
        rightSide_44P[:, :, k] = inv(invRightSide_44P[:, :, k])

    Ak_perPatch_34P = np.matmul(leftSide_34P, rightSide_44P)

    return Ak_perPatch_34P

