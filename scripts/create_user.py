"""Compatibility shim: previous demo script was intentionally disabled.

Administrator creation is permitted ONLY in the authenticated super-admin GUI,
which verifies the physical USB key and active super-admin session.
"""
raise SystemExit(
    'SPOT security update: unrestricted account creation is disabled. '
    'Log in with the provisioned super-admin USB and choose Create Administrator.'
)
