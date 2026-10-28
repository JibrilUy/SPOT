"""Trusted one-time bootstrap: run as database owner on the SPOT PC.

Example: python -m scripts.provision_superadmin --drive E:\\ --username superadmin
The plugged USB must be reported as 'Removable' by Windows.
"""
import argparse
from pathlib import Path
from database.db import Database
from security.usb_key import write_new_token


def main():
    parser = argparse.ArgumentParser(description='Bootstrap FIRST SPOT super admin USB')
    parser.add_argument('--drive', required=True, help='Removable USB root, e.g. E:\\')
    parser.add_argument('--username', default='superadmin')
    args = parser.parse_args()

    print('This is a one-time setup. Do not run it on an untrusted computer.')
    answer = input('Type PROVISION to create the one and only super-admin USB key: ')
    if answer != 'PROVISION':
        raise SystemExit('Canceled.')

    db = Database()
    key_path = None
    try:
        key_path, key_id, secret = write_new_token(args.drive)
        try:
            user_id = db.provision_first_superadmin(args.username, key_id, secret)
        except Exception:
            key_path.unlink(missing_ok=True)
            raise
        print(f'Provisioned super admin #{user_id}. Keep this USB safe: {key_path}')
        print('The plain USB file is copyable; use only for a supervised prototype.')
    finally:
        db.close()


if __name__ == '__main__':
    main()
