import numpy as np


class EstimateInvBkPerPatch:
    def __init__(self, imgInput_4R, imgMatch_4R, linearIndicesInWindow_NP):
        self.imgInput_4R = imgInput_4R
        self.imgMatch_4R = imgMatch_4R
        self.linearIndicesInWindow_NP = linearIndicesInWindow_NP

    def estimate_invBk_perPatch(self, epsilon, gamma):
        # Gather matrices vk_bar(I), vk_bar(M), vk(T), for each patch
        Iban_4NP = np.reshape(self.imgInput_4R[:, self.linearIndicesInWindow_NP], [4, -1], order='F')
        Mar_4NP = np.reshape(self.imgMatch_4R[:, self.linearIndicesInWindow_NP], [4, -1], order='F')

        # We obtain (Bk)^-1 at each patch by vectorizing:
        # invBk_44 = Ibar*Ibar' + epsilon*(Mbar*Mbar') + gamma*eye(4);
        Iban_Iban_t_44P = np.dot(Iban_4NP, Iban_4NP.T)
        Mar_Mar_t_44P = np.dot(Mar_4NP, Mar_4NP.T)
        invBk_perPatch_44P = np.tile(gamma * np.eye(4), (1, Iban_Iban_t_44P.shape[1], 1))
        invBk_perPatch_44P += np.multiply(Iban_Iban_t_44P, epsilon)
        invBk_perPatch_44P += np.multiply(Mar_Mar_t_44P, epsilon)

        return invBk_perPatch_44P

