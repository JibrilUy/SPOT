"""Install cross-floor tracking flags without replacing the GUI controller.

From the SPOT project root, AFTER copying the core/ files:
    py scripts/enable_cross_floor_tracking.py

The script validates both changes before creating .bak_crossfloor backups.
It intentionally makes no changes to floor page UI, PostgreSQL, or login.
"""
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def patch_main(text):
    if "from core.shared_target_registry import SharedTargetRegistry" not in text:
        match = re.search(r"(?m)^from PyQt6 import [^\n]*QtWidgets[^\n]*$", text)
        if match is None:
            raise ValueError("main.py: can't find the PyQt6 QtWidgets import")
        text = text[:match.end()] + ("\nfrom core.shared_target_registry "
                                   "import SharedTargetRegistry") + text[match.end():]

    if "self.shared_target = SharedTargetRegistry()" not in text:
        anchor = "        self._build_floor_cameras()"
        if text.count(anchor) != 1:
            raise ValueError("main.py: can't find exactly one _build_floor_cameras() call")
        text = text.replace(anchor,
                            "        self.shared_target = SharedTargetRegistry()\n" + anchor, 1)

    start = text.find("floor_camera_sources = {")
    if start < 0:
        raise ValueError("main.py: can't find floor_camera_sources mapping")
    # Only edit the floor camera source block: don't accidentally modify FLOOR_PAGES.
    end = text.find("\n        for floor_num, config in floor_camera_sources.items():", start)
    if end < 0:
        raise ValueError("main.py: can't identify the end of floor_camera_sources")
    fragment = text[start:end]
    for floor in range(1, 5):
        expression = re.compile(rf"(?m)^(\s*{floor}:\s*\{{[^\n{{}}]*)(\}})(\s*,?[^\n]*)$")
        match = expression.search(fragment)
        if not match:
            raise ValueError(f"main.py: Floor {floor} source must be a one-line dictionary")
        content = match.group(1)
        additions = []
        flag = "secure_zone_tracking" if floor == 1 else "cross_floor_tracking"
        if not re.search(rf"['\"]{flag}['\"]\s*:\s*True", content):
            if re.search(rf"['\"]{flag}['\"]\s*:", content):
                raise ValueError(f"main.py: Floor {floor} {flag} exists but isn't True")
            additions.append(f'"{flag}": True')
        if not re.search(r"['\"]shared_target['\"]\s*:", content):
            additions.append('"shared_target": self.shared_target')
        if additions:
            insertion = (", " if not content.rstrip().endswith("{") else "") + ", ".join(additions)
            fragment = fragment[:match.end(1)] + insertion + fragment[match.end(1):]
    return text[:start] + fragment + text[end:]


def patch_controller(text):
    if "cross_floor_tracking=False" not in text:
        anchor = "        secure_zone_tracking=False,"
        if text.count(anchor) != 1:
            raise ValueError("floor_camera_controller.py: expected secure_zone_tracking=False once")
        text = text.replace(anchor, anchor +
                            "\n        cross_floor_tracking=False,\n        shared_target=None,", 1)

    if "self._cross_floor_tracking = cross_floor_tracking" not in text:
        anchor = "        self._secure_zone_tracking = secure_zone_tracking"
        if text.count(anchor) != 1:
            raise ValueError("floor_camera_controller.py: can't find _secure_zone_tracking assignment")
        text = text.replace(anchor, anchor +
                            "\n        self._cross_floor_tracking = cross_floor_tracking", 1)

    old_check = "        if enable_tracking and secure_zone_tracking:\n"
    new_check = "        if sum((bool(enable_tracking), bool(secure_zone_tracking), bool(cross_floor_tracking))) > 1:\n"
    if old_check in text:
        text = text.replace(old_check, new_check, 1)
    old_msg = 'raise ValueError("Choose manual tracking OR automatic secure-zone tracking.")'
    if old_msg in text:
        text = text.replace(old_msg,
                            'raise ValueError("Choose only one tracking mode per floor.")', 1)

    old_worker = "                camera_index=source, model_path=model_path, loop=loop, zone=zone\n"
    new_worker = ("                camera_index=source, model_path=model_path, loop=loop, zone=zone,\n"
                  "                shared_target=shared_target, floor_num=floor_num\n")
    if old_worker in text:
        text = text.replace(old_worker, new_worker, 1)
    elif new_worker not in text:
        raise ValueError("floor_camera_controller.py: can't locate Floor 1 SecureZoneWorker arguments")

    new_branch = '''        elif cross_floor_tracking:
            # No drawing controls or video overlays. Floor 1 supplies the
            # white-shirt appearance reference; this feed owns its own trail.
            self.camera_worker = SecureZoneWorker(
                camera_index=source, model_path=model_path, loop=loop, zone=None,
                shared_target=shared_target, floor_num=floor_num,
                follow_reference=True,
            )
            print(f"[SPOT Cross-floor] Floor {floor_num} waiting for Floor 1 white-shirt target")
'''
    if "        elif cross_floor_tracking:" not in text:
        anchor = "        else:\n            track_button = getattr(ui, track_button_attr, None) if enable_tracking else None"
        if text.count(anchor) != 1:
            raise ValueError("floor_camera_controller.py: manual tracker branch differs from expected version")
        text = text.replace(anchor, new_branch + anchor, 1)
    return text


def main():
    files = [(ROOT / "main.py", patch_main),
             (ROOT / "controllers" / "floor_camera_controller.py", patch_controller)]
    changes = []
    for path, patcher in files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file is missing: {path}")
        before = path.read_text(encoding="utf-8")
        after = patcher(before)
        compile(after, str(path), "exec")
        changes.append((path, before, after))

    written = []
    try:
        for path, before, after in changes:
            if before == after:
                print("UNCHANGED:", path.relative_to(ROOT), "(already patched)")
                continue
            backup = path.with_name(path.name + ".bak_crossfloor")
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(after, encoding="utf-8")
            written.append((path, before))
            print("PATCHED:", path.relative_to(ROOT), "backup:", backup.name)
    except OSError:
        for path, before in reversed(written):
            path.write_text(before, encoding="utf-8")
        raise
    print("Complete. Restart with: py main.py")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, SyntaxError) as exc:
        print("CROSS-FLOOR PATCH FAILED:", exc, file=sys.stderr)
        sys.exit(1)
