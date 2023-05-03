import numpy as np
import LocalTransfer


class LocalTransfer_affineModel(LocalTransfer):

    def __init__(self, imgInput, imgMatch, imgTarget):
        super().__init__(imgInput, imgMatch, imgTarget)
        R = self.imgSize[0] * self.imgSize[1]
        self.imgInput_4R = np.vstack((self.imgInput_3R, np.ones((1, R))))
        self.imgMatch_4R = np.vstack((self.imgMatch_3R, np.ones((1, R))))
        self.imgTarget_4R = np.vstack((self.imgTarget_3R, np.ones((1, R))))

