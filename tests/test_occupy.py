import unittest
from src.gpu_occupier.occupy import GPUOccupier, MultiGPUOccupier

class TestGPUOccupier(unittest.TestCase):
    def setUp(self):
        self.gpu_occupier = GPUOccupier(gpu_id=0, threshold=20, check_interval=1, occupation_ratio=0.7)

    def test_initialization(self):
        self.assertEqual(self.gpu_occupier.gpu_id, 0)
        self.assertEqual(self.gpu_occupier.threshold, 20)
        self.assertEqual(self.gpu_occupier.check_interval, 1)
        self.assertEqual(self.gpu_occupier.occupation_ratio, 0.7)

    def test_get_gpu_memory_usage(self):
        usage, total_memory = self.gpu_occupier.get_gpu_memory_usage()
        self.assertIsInstance(usage, float)
        self.assertIsInstance(total_memory, float)

class TestMultiGPUOccupier(unittest.TestCase):
    def setUp(self):
        self.multi_gpu_occupier = MultiGPUOccupier(gpu_ids=[0, 1], threshold=20)

    def test_initialization(self):
        self.assertEqual(len(self.multi_gpu_occupier.gpu_occupiers), 2)

    def test_cleanup(self):
        self.multi_gpu_occupier.cleanup()
        self.assertFalse(self.multi_gpu_occupier.running)

if __name__ == '__main__':
    unittest.main()