import numpy as np
from scipy.sparse import coo_matrix


def buildClosedFormMatrices(obj, G_34, epsilon, gamma):
    N = obj.linearIndicesInWindow_NP.shape[0]  # number of pixels per patch
    P = obj.linearIndicesInWindow_NP.shape[1]  # number of patches
    R = obj.imgInput_3R.shape[1]  # total number of pixels

    # Gather matrices vk_bar(I), vk_bar(M), vk(T), for each patch
    Iban_4NP = obj.imgInput_4R[:, obj.linearIndicesInWindow_NP.flatten()].reshape(4, N, P)
    Mar_4NP = obj.imgMatch_4R[:, obj.linearIndicesInWindow_NP.flatten()].reshape(4, N, P)
    T_3NP = obj.imgTarget_3R[:, obj.linearIndicesInWindow_NP.flatten()].reshape(3, N, P)

    # Compute inverse matrix Bk at each patch
    invBk_perPatch_44P = obj.estimateInvBk_perPatch(epsilon, gamma)
    Bk_44P = np.linalg.inv(invBk_perPatch_44P)

    # We obtain the matrix values for each patch by:
    # sparseMatrix_vals_NN = eye(N) - Ibar' * Bk * Ibar;
    Bk_Iban_4NP = np.matmul(Bk_44P, Iban_4NP)
    Iban_t_Bk_Iban_NNP = np.matmul(Iban_4NP.transpose(0, 2, 1), Bk_Iban_4NP)
    sparseMatrix_vals_NNP = np.eye(N) - Iban_t_Bk_Iban_NNP

    # We obtain the right-side vector elements for each patch by:
    # rightSide_vals_3N = (epsilon * T*Mbar' + gamma*G) * Bk * Ibar;
    T_Mar_t_34P = np.matmul(T_3NP, Mar_4NP.transpose(0, 2, 1))
    rightSide_vals_3NP = np.matmul((epsilon * T_Mar_t_34P + gamma * G_34), Bk_Iban_4NP)

    # Store the sparse matrix triplets
    sparseMatrix_rows_NNP = obj.linearIndicesInWindow_NP[obj.sparseLinearIndices_pattern_row, :, :]
    sparseMatrix_cols_NNP = obj.linearIndicesInWindow_NP[obj.sparseLinearIndices_pattern_row[:, np.newaxis], :, :]
    sparseMatrix_rows_NP = np.repeat(sparseMatrix_rows_NNP, N, axis=1)
    sparseMatrix_cols_NP = np.tile(sparseMatrix_cols_NNP, (N, 1, 1))
    sparseMatrix_vals_NP = sparseMatrix_vals_NNP.transpose(2, 0, 1).flatten()

    # Assemble the final sparse matrix
    M_sparseRR = coo_matrix((sparseMatrix_vals_NP, (sparseMatrix_rows_NP.flatten(), sparseMatrix_cols_NP.flatten())),
                            shape=(R, R)).tocsr()

    # Assemble the right side
    # TODO: replace this with an accumarray
    u_3R = np.zeros((3, R))
    for k in range(P):
        u_3R[:, obj.linearIndicesInWindow_NP[:, k]] += rightSide_vals_3NP[:, :, k]

    return M_sparseRR, u_3R, invBk_perPatch_44P
