import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "memos"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
