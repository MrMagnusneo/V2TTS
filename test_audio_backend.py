import sys
import builtins
import unittest
from unittest.mock import patch

from audio_backend import get_sounddevice, get_soundfile

class TestAudioBackend(unittest.TestCase):
    def test_get_sounddevice_success(self):
        # By default, trying to import sounddevice should succeed or fail depending on env,
        # but let's mock it to succeed and return a dummy object
        dummy_module = object()
        with patch.dict('sys.modules', {'sounddevice': dummy_module}):
            self.assertEqual(get_sounddevice(), dummy_module)

    def test_get_sounddevice_oserror(self):
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'sounddevice':
                raise OSError("mock oserror")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            if 'sounddevice' in sys.modules:
                del sys.modules['sounddevice']
            with self.assertRaises(RuntimeError) as cm:
                get_sounddevice()
            self.assertTrue(str(cm.exception).startswith("PortAudio library not found."))

    def test_get_sounddevice_other_error(self):
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'sounddevice':
                raise ImportError("mock import error")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            if 'sounddevice' in sys.modules:
                del sys.modules['sounddevice']
            with self.assertRaises(RuntimeError) as cm:
                get_sounddevice()
            self.assertTrue(str(cm.exception).startswith("Failed to import sounddevice: mock import error"))

    def test_get_soundfile_success(self):
        dummy_module = object()
        with patch.dict('sys.modules', {'soundfile': dummy_module}):
            self.assertEqual(get_soundfile(), dummy_module)

    def test_get_soundfile_error(self):
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == 'soundfile':
                raise ImportError("mock import error")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            if 'soundfile' in sys.modules:
                del sys.modules['soundfile']
            with self.assertRaises(RuntimeError) as cm:
                get_soundfile()
            self.assertTrue(str(cm.exception).startswith("Failed to import soundfile: mock import error"))

if __name__ == '__main__':
    unittest.main()
