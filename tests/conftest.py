import sys
from pathlib import Path

# Cho phép `import core`, `import retrieval`, ... trực tiếp từ src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
