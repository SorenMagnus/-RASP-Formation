"""测试公共初始化。"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    """将 src 目录加入测试进程的导入路径。"""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()
