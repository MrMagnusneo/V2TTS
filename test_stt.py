import unittest
from stt import default_compute_type

class TestDefaultComputeType(unittest.TestCase):
    def test_cuda_device(self):
        self.assertEqual(default_compute_type('cuda'), 'float16')

    def test_cpu_device(self):
        self.assertEqual(default_compute_type('cpu'), 'int8')

    def test_fallback_devices(self):
        self.assertEqual(default_compute_type('mps'), 'int8')
        self.assertEqual(default_compute_type('unknown'), 'int8')
        self.assertEqual(default_compute_type(''), 'int8')

if __name__ == '__main__':
    unittest.main()
