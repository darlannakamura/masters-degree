import os, unittest

from denoising.methods.neural_network.deep_image_prior import DeepImagePrior

class TestDeepImagePrior(unittest.TestCase):
    def test_improvement_with_deep_image_prior(self):
        DeepImagePrior(iterations=100, 
            filename='/media/darlan/DATA3/deeplearning-projects/prisma-library/denoising/methods/neural_network/deep_image_prior/data/cameraman.png'
        )

if __name__ == '__main__':
    unittest.main()
