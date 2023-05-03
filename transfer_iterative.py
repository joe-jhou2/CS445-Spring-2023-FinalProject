

def transfer_iterative(obj, nbIterations):
    import numpy as np

    # Parameters
    epsilon = 1
    gamma = 0.01

    # Initialize cell arrays that will contain per-iteration results
    savePerIterationResults = (nargout > 3)
    if savePerIterationResults:
        imgOutput_perIteration = [None] * nbIterations
        localTransform_array3CR_perIteration = [None] * nbIterations

    # Compute global transform, and globally transformed image
    globalTransform_3C = obj.computeGlobalTransform()

    # Precompute Bk^-1 for each patch
    invBk_perPatch_CCP = obj.estimateInvBk_perPatch(epsilon, gamma)

    for it in range(nbIterations):

        if it == 0:
            # Initialize local transforms
            Ak_perPatch_3CP = obj.initializeLocalTransforms(
                globalTransform_3C, epsilon, gamma)
        else:
            # Re-estimate the local transforms, given the new output
            Ak_perPatch_3CP = obj.estimateLocalTransforms_givenOutput(
                invBk_perPatch_CCP, globalTransform_3C, epsilon, gamma)

        # Apply transform on each patch to estimate the output image
        obj.estimateOutput_givenLocalTransforms(Ak_perPatch_3CP)

        if savePerIterationResults or (it == nbIterations - 1):

            # Output transformed image
            imgOutput = np.reshape(obj.imgOutput_3R.T, obj.imgSize, order='F')

            # Output the per-pixel local transform and corresponding binary mask
            localTransform_array3CR, imgMask_HW = obj.perPixelLocalTransforms(
                Ak_perPatch_3CP)

            # Also save the per-iteration results
            if savePerIterationResults:
                imgOutput_perIteration[it] = imgOutput
                localTransform_array3CR_perIteration[it] = localTransform_array3CR

    if savePerIterationResults:
        return (imgOutput, localTransform_array3CR, imgMask_HW,
                imgOutput_perIteration, localTransform_array3CR_perIteration)
    else:
        return (imgOutput, localTransform_array3CR, imgMask_HW)


