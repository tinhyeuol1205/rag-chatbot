import os
import sys
from pathlib import Path

# Unit tests must not depend on a developer's .env or a real provider secret.
# Settings validates provider credentials during construction, so provide
# harmless test-only credentials before any test module imports core.config.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-only-key")
os.environ.setdefault("GEMINI_API_KEY", "test-only-key")

# Cho phép `import core`, `import retrieval`, ... trực tiếp từ src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
