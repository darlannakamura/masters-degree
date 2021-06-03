import os
import unittest
from loaders.data_loader import DataLoader
from .config_loader import load_config
from settings import BASE_DIR

class TestDataLoader(unittest.TestCase):
    def test_bsd300_loader(self):
        config = load_config(os.path.join(BASE_DIR, 'loaders', 'assets', 'bsd300.yml'))
        data_loader = DataLoader(config=config)

        x_train, _ = data_loader.get_train()
        x_test, _ = data_loader.get_test()

        self.assertEqual(x_train.shape[0], config['size']['train'])
        self.assertEqual(x_test.shape[0], config['size']['test'])

if __name__ == '__main__':
    unittest.main()
