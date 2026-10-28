"""Authorization unit tests using a fake DB connection (not real PostgreSQL).

Production end-to-end PostgreSQL/Windows/PyQt6 testing still required.
"""
from contextlib import nullcontext
import importlib
from types import ModuleType
import sys
import unittest
from unittest.mock import patch
import uuid


def load_db_module():
    try:
        return importlib.import_module('database.db')
    except ModuleNotFoundError as e:
        if e.name not in ('psycopg', 'psycopg.rows', 'psycopg_pool'):
            raise
        pg = ModuleType('psycopg')
        rows = ModuleType('psycopg.rows')
        rows.dict_row = object()
        pool = ModuleType('psycopg_pool')
        pool.ConnectionPool = object()
        with patch.dict(sys.modules, {
            'psycopg': pg, 'psycopg.rows': rows, 'psycopg_pool': pool
        }):
            return importlib.import_module('database.db')


class Cursor:
    def __init__(self, result=None):
        self.result = result

    def fetchone(self):
        return self.result


class FakeConn:
    def __init__(self, session_row):
        self.sql = []
        self.session_row = session_row

    def execute(self, sql, params=None):
        self.sql.append(sql)
        if 'SELECT u.id AS user_id FROM login_sessions' in sql:
            return Cursor(self.session_row)
        return Cursor(None)

    def transaction(self):
        return nullcontext()


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def connection(self):
        from contextlib import nullcontext
        return nullcontext(self.conn)


class AuthTests(unittest.TestCase):
    def test_admin_cannot_create_admin(self):
        module = load_db_module()
        conn = FakeConn(session_row=None)  # a non-superadmin session
        db = module.Database.__new__(module.Database)
        db.pool = FakePool(conn)
        db.hasher = type('Hasher', (), {'hash': lambda self, pwd: 'argon2hash'})()
        with self.assertRaises(PermissionError):
            db.create_admin(5, uuid.uuid4(), b'a' * 32, 'newadmin', 'very-long-password')
        self.assertFalse(any('INSERT INTO app_users' in q for q in conn.sql))

    def test_valid_session_id_but_missing_usb_secret_cannot_create_account(self):
        module = load_db_module()
        conn = FakeConn(session_row={'user_id': 1})
        db = module.Database.__new__(module.Database)
        db.pool = FakePool(conn)
        db.hasher = type('Hasher', (), {'hash': lambda self, pwd: 'argon2hash'})()
        with self.assertRaises(PermissionError):
            db.create_admin(5, uuid.uuid4(), b'b' * 32, 'newadmin', 'very-long-password')
        self.assertFalse(any('INSERT INTO app_users' in q for q in conn.sql))


if __name__ == '__main__':
    unittest.main()
