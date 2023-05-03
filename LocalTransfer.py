import numpy as np
from ApplyPrecomputedTransform import ApplyPrecomputedTransform



class LocalTransfer:

    def __init__(self, imgInput, imgMatch, imgTarget):
        assert imgInput is not None, "imgInput must not be empty"

        self.imgSize = imgInput.shape

        # Vectorize all images
        self.imgInput_3R = np.reshape(imgInput, [-1, np.prod(self.imgSize)])
        self.imgMatch_3R = np.reshape(imgMatch, [-1, np.prod(self.imgSize)])
        self.imgTarget_3R = np.reshape(imgTarget, [-1, np.prod(self.imgSize)])

        self.linearIndicesInWindow_NP = None
        self.sparseLinearIndices_pattern_row = None

    @staticmethod
    def ApplyPrecomputedTransform(imgInput, localTransform_array3CR, patchWidth):
        pass

    def computeGlobalTransform(self):
        pass

    def initializeLocalTransforms(self):
        pass

    def estimateLocalTransforms_givenOutput(self):
        pass

    def buildClosedFormMatrices(self):
        pass

    def estimateInvBk_perPatch(self):
        pass

    def estimateOutput_givenLocalTransforms(self):
        pass
