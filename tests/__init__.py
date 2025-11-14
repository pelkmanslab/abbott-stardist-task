import json
from pathlib import Path

import abbott_stardist_task

PACKAGE_DIR = Path(abbott_stardist_task.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)
