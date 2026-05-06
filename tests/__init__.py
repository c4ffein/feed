import sys
from pathlib import Path

# Make `import feed` work whether tests are run via `python -m unittest discover`
# from the project root or invoked directly.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
