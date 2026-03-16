"""
One-time data preparation for autoresearch experiments.

Downloads parquet shards, trains a tokenizer, and caches train-time kernel
assets into a shared offline cache tree.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken
import torch
from huggingface_hub import snapshot_download

from runtime_config import (
    DEFAULT_BASE_URL,
    DEFAULT_VOCAB_SIZE,
    CacheConfig,
    kernel_manifest_path,
    kernel_repo_path,
    package_name_from_repo_id,
    parse_prepare_config,
    resolve_cache_config,
)

# ---------------------------------------------------------------------------
# Constants (fixed evaluation harness)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = DEFAULT_VOCAB_SIZE
NUM_TRAIN_SHARDS = 10
DOWNLOAD_WORKERS = 8

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^
\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[
]*|\s*[
]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


def _default_cache() -> CacheConfig:
    return resolve_cache_config()


def download_single_shard(index: int, data_dir: str, base_url: str) -> bool:
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(data_dir, filename)
    temp_path = filepath + ".tmp"
    if os.path.exists(filepath):
        return True

    url = f"{base_url}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            os.replace(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, OSError) as exc:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {exc}")
            for path in (temp_path, filepath):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2**attempt)
    return False


def download_data(num_shards: int, data_dir: str, base_url: str, download_workers: int = 8) -> None:
    os.makedirs(data_dir, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    existing = sum(1 for idx in ids if os.path.exists(os.path.join(data_dir, f"shard_{idx:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {data_dir}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.starmap(download_single_shard, [(idx, data_dir, base_url) for idx in ids])

    ok = sum(1 for result in results if result)
    print(f"Data: {ok}/{len(ids)} shards ready at {data_dir}")


def list_parquet_files(data_dir: str | None = None) -> list[str]:
    data_dir = data_dir or _default_cache().data_dir
    if not os.path.isdir(data_dir):
        return []
    files = sorted(filename for filename in os.listdir(data_dir) if filename.endswith(".parquet") and not filename.endswith(".tmp"))
    return [os.path.join(data_dir, filename) for filename in files]


def text_iterator(data_dir: str, max_chars: int = 1_000_000_000, doc_cap: int = 10_000):
    parquet_paths = [path for path in list_parquet_files(data_dir) if not path.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        parquet_file = pq.ParquetFile(filepath)
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx)
            for text in row_group.column("text").to_pylist():
                document = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(document)
                yield document
                if nchars >= max_chars:
                    return


def train_tokenizer(data_dir: str, tokenizer_dir: str, vocab_size: int = VOCAB_SIZE) -> None:
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {tokenizer_dir}")
        return

    os.makedirs(tokenizer_dir, exist_ok=True)

    parquet_files = list_parquet_files(data_dir)
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(data_dir), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(key): value for key, value in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + idx for idx, name in enumerate(SPECIAL_TOKENS)}
    encoding = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as handle:
        pickle.dump(encoding, handle)

    print(f"Tokenizer: trained in {time.time() - t0:.1f}s, saved to {tokenizer_pkl}")

    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(encoding.n_vocab):
        token_str = encoding.decode([token_id])
        token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = encoding.encode_ordinary(test)
    decoded = encoding.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={encoding.n_vocab})")


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir: str | None = None):
        tokenizer_dir = tokenizer_dir or _default_cache().tokenizer_dir
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as handle:
            enc = pickle.load(handle)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads: int = 8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(tokenizer_dir: str | None = None, device: str | torch.device = "cpu"):
    tokenizer_dir = tokenizer_dir or _default_cache().tokenizer_dir
    path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(path, "rb") as handle:
        return torch.load(handle, map_location=device)


def _document_batches(
    split: str,
    data_dir: str | None = None,
    tokenizer_batch_size: int = 128,
    rank: int = 0,
    world_size: int = 1,
):
    data_dir = data_dir or _default_cache().data_dir
    parquet_paths = list_parquet_files(data_dir)
    assert parquet_paths, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(data_dir, VAL_FILENAME)
    if split == "train":
        parquet_paths = [path for path in parquet_paths if path != val_path]
        assert parquet_paths, "No training shards found."
    else:
        parquet_paths = [val_path]

    epoch = 1
    batch_idx = 0
    while True:
        for filepath in parquet_paths:
            parquet_file = pq.ParquetFile(filepath)
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(row_group_idx)
                batch = row_group.column("text").to_pylist()
                for start in range(0, len(batch), tokenizer_batch_size):
                    docs = batch[start:start + tokenizer_batch_size]
                    if split == "train" and batch_idx % world_size != rank:
                        batch_idx += 1
                        continue
                    yield docs, epoch
                    batch_idx += 1
        epoch += 1


def make_dataloader(
    tokenizer,
    B: int,
    T: int,
    split: str,
    data_dir: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    buffer_size: int = 1000,
    device: str | torch.device = "cuda",
):
    assert split in ["train", "val"]
    device = torch.device(device)
    row_capacity = T + 1
    batches = _document_batches(split, data_dir=data_dir, rank=rank, world_size=world_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=device.type == "cuda")
    device_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = device_buffer[:B * T].view(B, T)
    targets = device_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for idx, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = idx
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda idx: len(doc_buffer[idx]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        device_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(
    model,
    tokenizer,
    batch_size: int,
    data_dir: str | None = None,
    tokenizer_dir: str | None = None,
    device: str | torch.device = "cuda",
):
    device = torch.device(device)
    token_bytes = get_token_bytes(tokenizer_dir=tokenizer_dir, device=device)
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val", data_dir=data_dir, device=device)
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


def prefetch_kernel_assets(cache: CacheConfig, repo_ids: tuple[str, ...]) -> None:
    os.makedirs(cache.kernel_dir, exist_ok=True)
    manifest = {}
    for repo_id in repo_ids:
        target_dir = kernel_repo_path(cache, repo_id)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Kernels: caching {repo_id} -> {target_dir}")
        snapshot_download(
            repo_id,
            cache_dir=cache.cache_root,
            local_dir=target_dir,
            allow_patterns="build/*",
        )
        build_dir = os.path.join(target_dir, "build")
        if not os.path.isdir(build_dir):
            raise FileNotFoundError(f"Kernel repo {repo_id} did not materialize a build/ directory at {target_dir}")
        manifest[repo_id] = {
            "path": target_dir,
            "package_name": package_name_from_repo_id(repo_id),
        }

    manifest_path = kernel_manifest_path(cache)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"Kernels: wrote manifest to {manifest_path}")


def main(argv: list[str] | None = None) -> None:
    cfg = parse_prepare_config(argv)
    num_shards = NUM_TRAIN_SHARDS

    print(f"Cache root: {cfg.cache.cache_root}")
    print(f"Data directory: {cfg.cache.data_dir}")
    print(f"Tokenizer directory: {cfg.cache.tokenizer_dir}")
    print(f"Kernel directory: {cfg.cache.kernel_dir}")
    print()

    download_data(num_shards, cfg.cache.data_dir, DEFAULT_BASE_URL, download_workers=DOWNLOAD_WORKERS)
    print()

    train_tokenizer(cfg.cache.data_dir, cfg.cache.tokenizer_dir, vocab_size=VOCAB_SIZE)
    print()

    if cfg.prefetch_kernels:
        prefetch_kernel_assets(cfg.cache, cfg.kernel_repos)
        print()
    else:
        print("Kernels: prefetch disabled")
        print()

    print("Done! Ready to train.")


if __name__ == "__main__":
    main()
