"""Windows removable-USB secret-file login for the SPOT single-PC prototype.

NOTE: This proves possession of a *copyable file*, NOT genuine physical USB
hardware. A cloned flash drive can authenticate. Upgrade to a FIDO2/security
key with hardware-backed challenge/response for non-prototype deployment.
"""
import base64
import binascii
import ctypes
import json
import os
from pathlib import Path
import secrets
import string
import sys
import uuid

TOKEN_FILENAME = '.spot-superadmin-key.json'
TOKEN_VERSION = 1


def is_windows_removable_drive(root):
    """A drive reported by Windows as DRIVE_REMOVABLE (USB sticks typically)."""
    if sys.platform != 'win32':
        return False
    p = Path(root)
    drive = p.drive
    if len(drive) != 2 or drive[1] != ':':
        return False
    drive_root = drive + '\\'
    return ctypes.windll.kernel32.GetDriveTypeW(drive_root) == 2


def removable_roots():
    """Scan only removable drives; NEVER use a user-supplied arbitrary path."""
    if sys.platform != 'win32':
        return []
    return [f'{letter}:\\' for letter in string.ascii_uppercase
            if is_windows_removable_drive(f'{letter}:\\')]


def read_token(drive_root):
    """Return (UUID, 32-byte secret) or None. Refuse non-removable volumes."""
    if not is_windows_removable_drive(drive_root):
        return None
    token_path = Path(drive_root) / TOKEN_FILENAME
    try:
        if token_path.stat().st_size > 4096:
            return None
        data = json.loads(token_path.read_text(encoding='utf-8'))
        if data.get('version') != TOKEN_VERSION:
            return None
        key_id = uuid.UUID(data['key_id'])
        secret = base64.b64decode(data['secret'], validate=True)
        if len(secret) != 32:
            return None
        return key_id, secret
    except (OSError, ValueError, TypeError, KeyError, AttributeError, binascii.Error):
        return None


def find_tokens():
    for root in removable_roots():
        token = read_token(root)
        if token is not None:
            yield token


def write_new_token(drive_root):
    """One-time key generation. Caller must remove file on DB failure."""
    if not is_windows_removable_drive(drive_root):
        raise ValueError('Use a Windows drive letter for a REMOVABLE USB drive')
    key_id = uuid.uuid4()
    secret = secrets.token_bytes(32)
    payload = {
        'version': TOKEN_VERSION,
        'key_id': str(key_id),
        'secret': base64.b64encode(secret).decode('ascii'),
    }
    path = Path(drive_root) / TOKEN_FILENAME
    # Exclusive creation ensures an existing USB key is never overwritten.
    with path.open('x', encoding='utf-8') as file:
        json.dump(payload, file)
    return path, key_id, secret
