"""All persistence and account authorization for single-PC SPOT.

Never expose generic 'create_user' or role-escalation methods to an unauthenticated
UI. The one-time bootstrap is separate from ordinary super-admin operations.
"""
import hashlib
import hmac
import os
import re
import uuid

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class Database:
    def __init__(self):
        load_dotenv()
        dsn = os.environ.get('DATABASE_URL')
        if not dsn:
            raise RuntimeError('DATABASE_URL is missing; see .env.example')
        self.hasher = PasswordHasher()
        self.pool = ConnectionPool(
            conninfo=dsn,
            kwargs={'row_factory': dict_row, 'connect_timeout': 5},
            min_size=1, max_size=4, timeout=5, open=True,
        )
        try:
            self.pool.wait(timeout=5)
        except Exception:
            self.pool.close()
            raise

    def close(self):
        self.pool.close()

    @staticmethod
    def _validate_username(username):
        username = username.strip().lower()
        if not re.fullmatch(r'[a-z0-9_.-]{3,64}', username):
            raise ValueError('Username needs 3-64 characters: a-z 0-9 _ . -')
        return username

    def provision_first_superadmin(self, username, key_id, secret):
        """ONE-TIME trusted-machine bootstrap, never called from SPOT GUI.

        Requires the DB owner connection and an empty super-admin slot.
        """
        username = self._validate_username(username)
        if not isinstance(secret, bytes) or len(secret) != 32:
            raise ValueError('The USB key needs 32 random secret bytes')
        key_id = uuid.UUID(str(key_id))
        with self.pool.connection() as conn:
            with conn.transaction():
                # Serialized bootstrap attempts, even from separate processes.
                conn.execute('LOCK TABLE app_users IN SHARE ROW EXCLUSIVE MODE')
                if conn.execute("SELECT 1 FROM app_users WHERE role='super_admin'").fetchone():
                    raise PermissionError('Super admin already provisioned')
                row = conn.execute(
                    """INSERT INTO app_users(username, role, password_hash)
                       VALUES (%s, 'super_admin', NULL) RETURNING id""",
                    (username,),
                ).fetchone()
                conn.execute(
                    'INSERT INTO usb_keys (key_id, user_id, secret_hash) VALUES (%s, %s, %s)',
                    (key_id, row['id'], hashlib.sha256(secret).digest()),
                )
                conn.execute(
                    "INSERT INTO audit_events (target_user_id, event_type) VALUES (%s, 'superadmin_provisioned')",
                    (row['id'],),
                )
        return row['id']

    def authenticate_admin(self, username, password):
        """Administrator-only password step; FaceAuthWorker does second step."""
        if not username or not password:
            return None
        username = username.strip().lower()
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT id, username, role, face_label, password_hash
                   FROM app_users
                   WHERE username=%s AND role='admin' AND is_active=TRUE""",
                (username,),
            ).fetchone()
        if not row:
            return None
        try:
            self.hasher.verify(row['password_hash'], password)
        except (VerificationError, InvalidHashError, ValueError):
            return None
        # Refuse admin login until enrollment is associated with an actual label.
        if row['face_label'] is None:
            return None
        return {k: row[k] for k in ('id', 'username', 'role', 'face_label')}

    def start_admin_session(self, admin_user_id):
        """Call ONLY after successful continuous face verification."""
        with self.pool.connection() as conn:
            row = conn.execute(
                """INSERT INTO login_sessions (user_id, auth_method)
                   SELECT id, 'password_face' FROM app_users
                   WHERE id=%s AND role='admin' AND is_active=TRUE
                   RETURNING id""",
                (admin_user_id,),
            ).fetchone()
        if row is None:
            raise PermissionError('Administrator disabled or unavailable')
        return row['id']

    def _find_active_key_account(self, conn, key_id, secret):
        try:
            key_id = uuid.UUID(str(key_id))
        except (ValueError, AttributeError, TypeError):
            return None
        if not isinstance(secret, bytes) or len(secret) != 32:
            return None
        row = conn.execute(
            """SELECT u.id, u.username, u.role, k.secret_hash
               FROM usb_keys k JOIN app_users u ON u.id=k.user_id
               WHERE k.key_id=%s AND k.revoked_at IS NULL
                 AND u.is_active=TRUE AND u.role='super_admin'""",
            (key_id,),
        ).fetchone()
        if row and hmac.compare_digest(bytes(row['secret_hash']), hashlib.sha256(secret).digest()):
            return row
        return None

    def start_superadmin_session(self, key_id, secret):
        """Validate the key's random secret before writing a session."""
        with self.pool.connection() as conn:
            with conn.transaction():
                row = self._find_active_key_account(conn, key_id, secret)
                if row is None:
                    return None
                session = conn.execute(
                    """INSERT INTO login_sessions (user_id, auth_method)
                       VALUES (%s, 'usb_key') RETURNING id""",
                    (row['id'],),
                ).fetchone()
                return ({'id': row['id'], 'username': row['username'], 'role': 'super_admin'}, session['id'])

    def usb_key_still_authorized(self, session_id, key_id, secret):
        """Recheck USB credential AND the live super-admin session."""
        with self.pool.connection() as conn:
            row = self._find_active_key_account(conn, key_id, secret)
            if row is None:
                return False
            active = conn.execute(
                """SELECT 1 FROM login_sessions s
                   WHERE s.id=%s AND s.user_id=%s
                   AND s.auth_method='usb_key' AND s.logged_out_at IS NULL""",
                (session_id, row['id']),
            ).fetchone()
            return active is not None

    def _require_super_admin(self, conn, session_id, key_id, secret):
        # Do not trust a stale/copied session ID. Every privileged DB call
        # must ALSO prove possession of the live USB key's random secret.
        verified = conn.execute(
            """SELECT u.id AS user_id FROM login_sessions s
               JOIN app_users u ON u.id=s.user_id
               WHERE s.id=%s AND s.logged_out_at IS NULL
                 AND s.auth_method='usb_key' AND u.is_active=TRUE
                 AND u.role='super_admin'
               FOR SHARE OF s, u""",
            (session_id,),
        ).fetchone()
        key_user = self._find_active_key_account(conn, key_id, secret)
        if not verified or not key_user or verified['user_id'] != key_user['id']:
            raise PermissionError('Active super-admin session AND valid USB key required')

    def create_admin(self, super_session_id, key_id, secret, username, password, full_name=None, face_label=None, *, is_active=True):
        """Super admin only. Admin cannot create accounts, even via GUI callbacks."""
        username = self._validate_username(username)
        if len(password) != 8:
            raise ValueError('Administrator password must contain exactly 8 characters')
        if face_label is not None and (not isinstance(face_label, int) or face_label < 0):
            raise ValueError('face_label must be a nonnegative integer or empty')
        pwd_hash = self.hasher.hash(password)
        with self.pool.connection() as conn:
            with conn.transaction():
                self._require_super_admin(conn, super_session_id, key_id, secret)
                row = conn.execute(
                    """INSERT INTO app_users(username, full_name, role, password_hash, face_label, is_active)
                       VALUES (%s, %s, 'admin', %s, %s, %s) RETURNING id""",
                    (username, full_name, pwd_hash, face_label, is_active),
                ).fetchone()
                conn.execute(
                    """INSERT INTO audit_events(actor_session_id, target_user_id, event_type)
                       VALUES (%s, %s, 'admin_created')""",
                    (super_session_id, row['id']),
                )
        return row['id']

    def activate_enrolled_admin(self, super_session_id, key_id, secret, admin_id):
        """Only the USB super admin can enable a completed face enrollment."""
        with self.pool.connection() as conn:
            with conn.transaction():
                self._require_super_admin(conn, super_session_id, key_id, secret)
                row = conn.execute(
                    """UPDATE app_users SET is_active=TRUE
                       WHERE id=%s AND role='admin' AND face_label IS NOT NULL
                       RETURNING id""", (admin_id,),
                ).fetchone()
                if not row:
                    raise PermissionError('No enrollable administrator found')
                conn.execute(
                    """INSERT INTO audit_events(actor_session_id, target_user_id, event_type)
                       VALUES (%s, %s, 'admin_face_enrolled')""",
                    (super_session_id, admin_id),
                )

    def change_admin_password(self, super_session_id, key_id, secret, admin_id, password):
        """USB-super-admin-only password reset, with audit event."""
        if len(password) != 8:
            raise ValueError('Administrator password must contain exactly 8 characters')
        password_hash = self.hasher.hash(password)
        with self.pool.connection() as conn:
            with conn.transaction():
                self._require_super_admin(conn, super_session_id, key_id, secret)
                target = conn.execute(
                    """UPDATE app_users SET password_hash=%s
                       WHERE id=%s AND role='admin' RETURNING id""",
                    (password_hash, admin_id),
                ).fetchone()
                if target is None:
                    raise ValueError('Administrator account not found')
                # Expire any old administrator sessions stored in the database.
                conn.execute(
                    """UPDATE login_sessions SET logged_out_at=NOW(),
                       logout_reason='system_error'
                       WHERE user_id=%s AND logged_out_at IS NULL""",
                    (admin_id,),
                )
                conn.execute(
                    """INSERT INTO audit_events(actor_session_id, target_user_id, event_type)
                       VALUES (%s, %s, 'admin_password_changed')""",
                    (super_session_id, admin_id),
                )

    def record_admin_face_updated(self, super_session_id, key_id, secret,
                                  admin_id, expected_face_label):
        """Audit an already-staged face model replacement.

        Requires fresh USB proof; fails if the chosen admin/label changed.
        Existing active sessions are ended in the database. The user must
        restart or log out of any already-open administrator UI as well.
        """
        with self.pool.connection() as conn:
            with conn.transaction():
                self._require_super_admin(conn, super_session_id, key_id, secret)
                target = conn.execute(
                    """SELECT id FROM app_users WHERE id=%s AND role='admin'
                       AND face_label=%s FOR UPDATE""",
                    (admin_id, expected_face_label),
                ).fetchone()
                if not target:
                    raise ValueError('Administrator face label changed; abort update.')
                conn.execute(
                    """UPDATE login_sessions SET logged_out_at=NOW(),
                       logout_reason='system_error'
                       WHERE user_id=%s AND logged_out_at IS NULL""",
                    (admin_id,),
                )
                conn.execute(
                    """INSERT INTO audit_events(actor_session_id, target_user_id, event_type)
                       VALUES (%s, %s, 'admin_face_updated')""",
                    (super_session_id, admin_id),
                )

    def list_admins(self, super_session_id, key_id, secret):
        with self.pool.connection() as conn:
            self._require_super_admin(conn, super_session_id, key_id, secret)
            return conn.execute(
                """SELECT u.id, u.username, u.full_name, u.face_label, u.is_active,
                          u.created_at, q.last_successful_login,
                          q.completed_session_seconds, q.successful_login_count
                   FROM app_users u
                   LEFT JOIN spot_user_login_summary q ON q.user_id=u.id
                   WHERE u.role='admin' ORDER BY u.created_at DESC, u.id DESC"""
            ).fetchall()

    def get_all_activity(self, super_session_id, key_id, secret, limit=200):
        """Super-admin-only view of sessions, monitoring page views, and admin creation."""
        limit = max(1, min(int(limit), 500))
        with self.pool.connection() as conn:
            self._require_super_admin(conn, super_session_id, key_id, secret)
            return conn.execute(
                """SELECT actor, event_type, detail, occurred_at FROM (
                    SELECT u.username AS actor, 'login' AS event_type,
                           s.auth_method AS detail, s.logged_in_at AS occurred_at
                    FROM login_sessions s JOIN app_users u ON u.id=s.user_id
                    UNION ALL
                    SELECT u.username, 'logout', s.auth_method, s.logged_out_at
                    FROM login_sessions s JOIN app_users u ON u.id=s.user_id
                    WHERE s.logged_out_at IS NOT NULL
                    UNION ALL
                    SELECT u.username, 'floor_page_open',
                           'Floor ' || v.floor_number::text, v.opened_at
                    FROM floor_page_visits v
                    JOIN login_sessions s ON s.id=v.session_id
                    JOIN app_users u ON u.id=s.user_id
                    UNION ALL
                    SELECT COALESCE(u.username, '[bootstrap]'), a.event_type,
                           COALESCE(t.username, ''), a.occurred_at
                    FROM audit_events a
                    LEFT JOIN login_sessions s ON s.id=a.actor_session_id
                    LEFT JOIN app_users u ON u.id=s.user_id
                    LEFT JOIN app_users t ON t.id=a.target_user_id
                    UNION ALL
                    SELECT l.entered_username, 'login_' || l.result,
                           l.auth_method || ' / ' || COALESCE(l.failure_reason,l.stage),
                           l.started_at
                    FROM login_attempts l
                    WHERE l.result <> 'successful'
                    UNION ALL
                    SELECT u.username, 'transcript_' || p.status,
                           p.transcript_date::text || ' / copies: ' || p.copies::text,
                           p.requested_at
                    FROM transcript_print_jobs p
                    JOIN login_sessions s ON s.id=p.requested_by_session_id
                    JOIN app_users u ON u.id=s.user_id
                ) a ORDER BY occurred_at DESC LIMIT %s""",
                (limit,),
            ).fetchall()

    # ---------------------------------------------------------------
    # Authentication attempt and printing audit (requires activity migration)
    # ---------------------------------------------------------------
    def start_login_attempt(self, entered_username, auth_method='password_face',
                            stage='password', user_id=None):
        """Write one record per LOGIN CLICK, never one per video frame."""
        entered_username = (entered_username or '').strip()[:128] or '[empty]'
        with self.pool.connection() as conn:
            row = conn.execute(
                """INSERT INTO login_attempts
                   (entered_username, user_id, auth_method, stage, result)
                   VALUES (%s, %s, %s, %s, 'in_progress') RETURNING id""",
                (entered_username, user_id, auth_method, stage),
            ).fetchone()
            return row['id']

    def update_login_attempt(self, attempt_id, *, stage=None, user_id=None):
        with self.pool.connection() as conn:
            if user_id is not None:
                conn.execute(
                    """UPDATE login_attempts SET stage=COALESCE(%s,stage), user_id=%s
                       WHERE id=%s AND result='in_progress'""",
                    (stage, user_id, attempt_id),
                )
            elif stage is not None:
                conn.execute(
                    """UPDATE login_attempts SET stage=%s
                       WHERE id=%s AND result='in_progress'""",
                    (stage, attempt_id),
                )

    def finish_login_attempt(self, attempt_id, result, *, failure_reason=None,
                             session_id=None):
        allowed = {'password_failed', 'face_failed', 'usb_failed',
                   'successful', 'cancelled', 'system_error'}
        if result not in allowed:
            raise ValueError('Invalid login attempt result')
        with self.pool.connection() as conn:
            conn.execute(
                """UPDATE login_attempts
                   SET result=%s, stage='complete', failure_reason=%s,
                       session_id=%s, finished_at=NOW()
                   WHERE id=%s AND result='in_progress'""",
                (result, (failure_reason or None), session_id, attempt_id),
            )

    def request_transcript_print(self, session_id, transcript_date, copies=1, printer_name=None):
        """Call right BEFORE sending the transcript to the printer."""
        with self.pool.connection() as conn:
            row = conn.execute(
                """INSERT INTO transcript_print_jobs
                   (requested_by_session_id, transcript_date, copies, printer_name)
                   SELECT id, %s, %s, %s FROM login_sessions
                   WHERE id=%s AND logged_out_at IS NULL
                   RETURNING id""",
                (transcript_date, copies, printer_name, session_id),
            ).fetchone()
        if row is None:
            raise PermissionError('No active session for transcript printing')
        return row['id']

    def finish_transcript_print(self, job_id, *, spooled=False, reason=None):
        """Spooling is NOT proof that paper was physically printed."""
        with self.pool.connection() as conn:
            conn.execute(
                """UPDATE transcript_print_jobs
                   SET status=%s, failure_reason=%s,
                       submitted_at=CASE WHEN %s THEN NOW() ELSE NULL END
                   WHERE id=%s AND status='requested'""",
                ('spooled' if spooled else 'failed', reason, spooled, job_id),
            )

    def record_floor_open(self, session_id, floor_number):
        with self.pool.connection() as conn:
            row = conn.execute(
                """INSERT INTO floor_page_visits(session_id, floor_number)
                   SELECT s.id, %s FROM login_sessions s
                   JOIN app_users u ON u.id=s.user_id
                   JOIN floors f ON f.floor_number=%s
                   WHERE s.id=%s AND s.logged_out_at IS NULL AND u.is_active=TRUE
                     AND u.role IN ('admin', 'super_admin')
                   RETURNING id""",
                (floor_number, floor_number, session_id),
            ).fetchone()
        return row is not None

    def end_session(self, session_id, reason='manual'):
        if reason not in ('manual', 'usb_removed', 'application_exit', 'system_error'):
            reason = 'manual'
        with self.pool.connection() as conn:
            conn.execute(
                """UPDATE login_sessions SET logged_out_at=NOW(), logout_reason=%s
                   WHERE id=%s AND logged_out_at IS NULL""",
                (reason, session_id),
            )
