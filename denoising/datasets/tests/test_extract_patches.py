import unittest
import numpy as np
from denoising.datasets import extract_patches

class TestExtractPatches(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.arr = np.random.rand(10, 500, 500, 1)

    def test_extract_patches(self):
        patches = extract_patches(self.arr, begin=(0,0), stride=10, dimension=(50,50), \
            quantity_per_image=(5,2))
        
        self.assertEqual(patches.shape[0], 10*5*2)

    def test_if_patches_has_four_dimensions(self):
        patches = extract_patches(self.arr, begin=(0,0), stride=10, dimension=(50,50), \
            quantity_per_image=(5,2))
        
        self.assertEqual(len(patches.shape), 4)

    def test_if_patches_images_has_specified_dimensions(self):
        patches = extract_patches(self.arr, begin=(0,0), stride=10, \
            dimension=(30,30), quantity_per_image=(5,2))
        
        self.assertEqual(patches.shape[1], 30)
        self.assertEqual(patches.shape[2], 30)
    
    def test_patches_starts(self):
        patches = extract_patches(self.arr, begin=(0,0), stride=10, \
            dimension=(30,30), quantity_per_image=(5,2))

        self.assertTrue((patches[0,:,:,:]==self.arr[0,:30,:30,:]).all())

    def test_stride(self):
        patches = extract_patches(self.arr, begin=(0,0), stride=10, \
            dimension=(30,30), quantity_per_image=(5,2))

        self.assertTrue((patches[1,:,:,:]==self.arr[0,:30,10:40,:]).all())

if __name__ == '__main__':
    unittest.main()
