"""Unit tests for token parsing and creation. Uses a temporary mock drive root.

Run: python -m unittest discover -s tests -v
No GUI or PostgreSQL needed for these tests.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from security import usb_key


class USBKeyTests(unittest.TestCase):
    def test_token_round_trip(self):
        with TemporaryDirectory() as tmp:
            with patch.object(usb_key, 'is_windows_removable_drive', return_value=True):
                path, key_id, secret = usb_key.write_new_token(tmp)
                self.assertEqual(path.name, usb_key.TOKEN_FILENAME)
                self.assertEqual(len(secret), 32)
                self.assertEqual(usb_key.read_token(tmp), (key_id, secret))
                with self.assertRaises(FileExistsError):
                    usb_key.write_new_token(tmp)

    def test_normal_disk_is_rejected(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(usb_key.read_token(tmp))
            with self.assertRaises(ValueError):
                usb_key.write_new_token(tmp)

    def test_invalid_token_is_rejected_without_crashing(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / usb_key.TOKEN_FILENAME
            p.write_text(json.dumps({'version': 1, 'key_id': 'not-a-uuid', 'secret': '???'}))
            with patch.object(usb_key, 'is_windows_removable_drive', return_value=True):
                self.assertIsNone(usb_key.read_token(tmp))

    def test_too_large_file_is_rejected(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / usb_key.TOKEN_FILENAME
            p.write_bytes(b'x' * 5000)
            with patch.object(usb_key, 'is_windows_removable_drive', return_value=True):
                self.assertIsNone(usb_key.read_token(tmp))


if __name__ == '__main__':
    unittest.main()
