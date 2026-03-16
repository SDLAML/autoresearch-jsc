from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
DEFAULT_VOCAB_SIZE = 8192
DEFAULT_NPROC_PER_NODE = 4
DEFAULT_FLASH_KERNEL_REPOS = (
    "varunneal/flash-attention-3",
    "kernels-community/flash-attn3",
)
KERNEL_MANIFEST_FILENAME = "manifest.json"


def _resolve_path(path: str | os.PathLike[str]) -> str:
    return str(Path(path).expanduser().resolve())


def default_cache_root() -> str:
    base_dir = os.environ.get("CACHE_DIR")
    if base_dir is None:
        base_dir = str(Path(__file__).resolve().parent / ".cache")
    return _resolve_path(Path(base_dir) / "autoresearch")


@dataclass(frozen=True)
class CacheConfig:
    cache_root: str
    data_dir: str
    tokenizer_dir: str
    kernel_dir: str


@dataclass(frozen=True)
class PrepareConfig:
    cache: CacheConfig
    prefetch_kernels: bool = True
    kernel_repos: tuple[str, ...] = DEFAULT_FLASH_KERNEL_REPOS


@dataclass(frozen=True)
class TrainConfig:
    cache: CacheConfig
    aspect_ratio: int = 96
    head_dim: int = 128
    window_pattern: str = "SSSL"
    total_batch_size: int = 2**18
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.04
    scalar_lr: float = 0.5
    weight_decay: float = 0.0
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0
    depth: int = 8
    device_batch_size: int = 32
    kv_heads: int = 0
    seed: int = 42
    nproc_per_node: int = DEFAULT_NPROC_PER_NODE
    flash_kernel_repo: str = "auto"
    no_spawn: bool = False


def resolve_cache_config(
    cache_root: str | None = None,
    data_dir: str | None = None,
    tokenizer_dir: str | None = None,
    kernel_dir: str | None = None,
) -> CacheConfig:
    resolved_root = _resolve_path(cache_root) if cache_root else default_cache_root()
    resolved_data_dir = _resolve_path(data_dir) if data_dir else _resolve_path(Path(resolved_root) / "data")
    resolved_tokenizer_dir = _resolve_path(tokenizer_dir) if tokenizer_dir else _resolve_path(Path(resolved_root) / "tokenizer")
    resolved_kernel_dir = _resolve_path(kernel_dir) if kernel_dir else _resolve_path(Path(resolved_root) / "kernels")
    return CacheConfig(
        cache_root=resolved_root,
        data_dir=resolved_data_dir,
        tokenizer_dir=resolved_tokenizer_dir,
        kernel_dir=resolved_kernel_dir,
    )


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def package_name_from_repo_id(repo_id: str) -> str:
    return repo_id.split("/")[-1].replace("-", "_")


def kernel_repo_path(cache: CacheConfig, repo_id: str) -> str:
    return _resolve_path(Path(cache.kernel_dir) / sanitize_repo_id(repo_id))


def kernel_manifest_path(cache: CacheConfig) -> str:
    return _resolve_path(Path(cache.kernel_dir) / KERNEL_MANIFEST_FILENAME)


def load_kernel_manifest(cache: CacheConfig) -> dict[str, dict[str, str]]:
    manifest_path = kernel_manifest_path(cache)
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _add_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-root", default=default_cache_root(), help="Root directory for all shared autoresearch caches.")
    parser.add_argument("--data-dir", default=None, help="Override the data directory. Defaults to <cache-root>/data.")
    parser.add_argument("--tokenizer-dir", default=None, help="Override the tokenizer directory. Defaults to <cache-root>/tokenizer.")
    parser.add_argument("--kernel-dir", default=None, help="Override the kernel asset directory. Defaults to <cache-root>/kernels.")


def build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare autoresearch data, tokenizer, and offline kernel assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_cache_args(parser)
    kernel_group = parser.add_mutually_exclusive_group()
    kernel_group.add_argument("--prefetch-kernels", dest="prefetch_kernels", action="store_true", help="Download train-time kernel assets into the shared cache.")
    kernel_group.add_argument("--no-prefetch-kernels", dest="prefetch_kernels", action="store_false", help="Skip kernel asset caching.")
    parser.set_defaults(prefetch_kernels=True)
    parser.add_argument(
        "--kernel-repo",
        dest="kernel_repos",
        action="append",
        default=None,
        help="Kernel repo to cache. Repeat to cache multiple repos. Defaults to both current Flash Attention repos.",
    )
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train autoresearch with CLI-configured single-node DDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_cache_args(parser)
    parser.add_argument("--aspect-ratio", type=int, default=96, help="Model width multiplier: model_dim ~= depth * aspect_ratio.")
    parser.add_argument("--head-dim", type=int, default=128, help="Target attention head dimension.")
    parser.add_argument("--window-pattern", default="SSSL", help="Sliding-window pattern per layer, using S and L.")
    parser.add_argument("--total-batch-size", type=int, default=2**18, help="Global token batch size per optimizer step across all ranks.")
    parser.add_argument("--embedding-lr", type=float, default=0.6, help="Embedding learning rate.")
    parser.add_argument("--unembedding-lr", type=float, default=0.004, help="LM head learning rate.")
    parser.add_argument("--matrix-lr", type=float, default=0.04, help="Muon learning rate for transformer block matrices.")
    parser.add_argument("--scalar-lr", type=float, default=0.5, help="Learning rate for scalar lambda parameters.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Muon cautious weight decay.")
    parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1.")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Fraction of the time budget used for LR warmup.")
    parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="Fraction of the time budget used for LR warmdown.")
    parser.add_argument("--final-lr-frac", type=float, default=0.0, help="Final learning rate multiplier after warmdown.")
    parser.add_argument("--depth", type=int, default=8, help="Number of transformer layers.")
    parser.add_argument("--device-batch-size", type=int, default=32, help="Per-rank batch size.")
    parser.add_argument("--kv-heads", type=int, default=0, help="Number of KV heads for GQA. 0 = same as n_heads (MHA).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed; rank id is added for distributed workers.")
    parser.add_argument("--nproc-per-node", type=int, default=DEFAULT_NPROC_PER_NODE, help="Number of local GPU worker processes.")
    parser.add_argument("--flash-kernel-repo", default="auto", help="Kernel repo to load. Use 'auto' to select based on CUDA capability.")
    parser.add_argument("--no-spawn", action="store_true", default=False, help=argparse.SUPPRESS)
    return parser


def parse_prepare_config(argv: list[str] | None = None) -> PrepareConfig:
    args = build_prepare_parser().parse_args(argv)
    cache = resolve_cache_config(
        cache_root=args.cache_root,
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        kernel_dir=args.kernel_dir,
    )
    kernel_repos = tuple(args.kernel_repos) if args.kernel_repos else DEFAULT_FLASH_KERNEL_REPOS
    return PrepareConfig(
        cache=cache,
        prefetch_kernels=args.prefetch_kernels,
        kernel_repos=kernel_repos,
    )


def parse_train_config(argv: list[str] | None = None) -> TrainConfig:
    args = build_train_parser().parse_args(argv)
    cache = resolve_cache_config(
        cache_root=args.cache_root,
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        kernel_dir=args.kernel_dir,
    )
    return TrainConfig(
        cache=cache,
        aspect_ratio=args.aspect_ratio,
        head_dim=args.head_dim,
        window_pattern=args.window_pattern,
        total_batch_size=args.total_batch_size,
        embedding_lr=args.embedding_lr,
        unembedding_lr=args.unembedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_ratio=args.warmup_ratio,
        warmdown_ratio=args.warmdown_ratio,
        final_lr_frac=args.final_lr_frac,
        depth=args.depth,
        device_batch_size=args.device_batch_size,
        kv_heads=args.kv_heads,
        seed=args.seed,
        nproc_per_node=args.nproc_per_node,
        flash_kernel_repo=args.flash_kernel_repo,
        no_spawn=args.no_spawn,
    )
