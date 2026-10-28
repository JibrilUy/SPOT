"""Safe one-time patch of the existing SPOT main.py and CameraWorker loop.

Run from SPOT project root:
    py scripts/enable_floor1_secure_zone.py

Never overwrites unrelated code; creates .bak_secure_zone backup before editing.
"""
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def patch_main(text):
    if '"secure_zone_tracking": True' in text or "'secure_zone_tracking': True" in text:
        return text, 'Floor 1 secure-zone flag is already present'
    # Narrow scope to floor_camera_sources mapping, not any other dictionary.
    start = text.find('floor_camera_sources = {')
    if start < 0:
        raise ValueError('main.py: Could not find floor_camera_sources = {')
    tail = text[start:]
    # Only match a single-line Floor 1 entry. Preserve its original keys/values.
    pattern = re.compile(r'(?m)^(\s*1\s*:\s*\{[^\n{}]*\})(\s*,?\s*)(#.*)?$')
    match = pattern.search(tail)
    if not match:
        raise ValueError('main.py: Floor 1 config not on one line; please add "secure_zone_tracking": True manually')
    old = match.group(1)
    new = old[:-1] + ', "secure_zone_tracking": True}'
    tail = tail[:match.start(1)] + new + tail[match.end(1):]
    return text[:start] + tail, 'Enabled secure-zone tracking in Floor 1 configuration'


def patch_camera_worker(text):
    if 'self._detector.reset_tracking()' in text:
        return text, 'Video-loop tracker reset is already present'
    matches = list(re.finditer(r'(?m)^(?P<indent>\s*)cap\.set\(cv2\.CAP_PROP_POS_FRAMES,\s*0\)[ \t]*$', text))
    if len(matches) != 1:
        raise ValueError('core/camera_worker.py: Expected exactly one cap.set(..., 0) video rewind')
    m = matches[0]
    indent = m.group('indent')
    code = ('\n' + indent + 'if hasattr(self._detector, "reset_tracking"):'
            '\n' + indent + '    self._detector.reset_tracking()')
    return text[:m.end()] + code + text[m.end():], 'Resets track IDs and trails when a video loops'


def apply(path, modifier):
    if not path.is_file():
        raise FileNotFoundError(f'Missing: {path}')
    original = path.read_text(encoding='utf-8')
    changed, info = modifier(original)
    if changed == original:
        print(f'UNCHANGED {path.relative_to(ROOT)}: {info}')
        return
    # Syntax validation before modifying the project.
    compile(changed, str(path), 'exec')
    backup = path.with_name(path.name + '.bak_secure_zone')
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(changed, encoding='utf-8')
    print(f'UPDATED {path.relative_to(ROOT)}: {info}')
    print(f'  backup: {backup.relative_to(ROOT)}')


if __name__ == '__main__':
    try:
        # Check both transformations before writing either file.
        main = ROOT / 'main.py'
        camera = ROOT / 'core' / 'camera_worker.py'
        patch_main(main.read_text(encoding='utf-8'))
        patch_camera_worker(camera.read_text(encoding='utf-8'))
        apply(main, patch_main)
        apply(camera, patch_camera_worker)
    except (ValueError, OSError, SyntaxError) as exc:
        print(f'PATCH FAILED: {exc}', file=sys.stderr)
        print('No additional changes made; see README.md for manual steps.', file=sys.stderr)
        sys.exit(1)
    print('Floor 1 integration enabled. Run: py main.py')
