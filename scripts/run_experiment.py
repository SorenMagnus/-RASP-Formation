"""批量实验命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    """将 src 目录注入导入路径，保证脚本可直接运行。"""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_bootstrap_src_path()

from apflf.sim.runner import run_batch  # noqa: E402
from apflf.utils.config import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="运行 APF-LF 最小可复现实验。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="实验随机种子列表。")
    parser.add_argument("--exp-id", default=None, help="可选的实验输出目录名。")
    return parser


def main() -> int:
    """执行实验并打印输出目录。"""

    parser = build_parser()
    args = parser.parse_args()
    config = load_config(Path(args.config))
    output_dir = run_batch(config=config, seeds=args.seeds, exp_id=args.exp_id)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
