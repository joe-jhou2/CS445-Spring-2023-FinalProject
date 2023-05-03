import numpy as np


class ComputeGlobalTransform:
    def __init__(self, imgMatch_3R, linearIndicesInWindow_NP, imgMatch_4R, imgTarget_3R, imgInput_3R):
        self.imgMatch_3R = imgMatch_3R
        self.linearIndicesInWindow_NP = linearIndicesInWindow_NP
        self.imgMatch_4R = imgMatch_4R
        self.imgTarget_3R = imgTarget_3R
        self.imgInput_3R = imgInput_3R

    def compute_global_transform(self):
        R = self.imgMatch_3R.shape[1]

        nonNaNPixels = np.unique(self.linearIndicesInWindow_NP)
        M_4R = self.imgMatch_4R[:, nonNaNPixels]
        T_3R = self.imgTarget_3R[:, nonNaNPixels]

        G_34 = np.dot(T_3R, np.transpose(M_4R)) / np.dot(M_4R, np.transpose(M_4R))

        if True:
            imgTransformed_3R = np.dot(G_34, np.vstack((self.imgInput_3R, np.ones((1, R)))))
            return G_34, imgTransformed_3R
