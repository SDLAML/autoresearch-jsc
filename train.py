"""
Autoresearch pretraining script with CLI-configured single-node DDP.
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from kernels import get_local_kernel
from torch.nn.parallel import DistributedDataParallel as DDP

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader
from runtime_config import DEFAULT_FLASH_KERNEL_REPOS, load_kernel_manifest, package_name_from_repo_id, parse_train_config

fa3 = None
H100_BF16_PEAK_FLOPS = 989.5e12


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    is_distributed: bool
    is_master: bool
    device: torch.device


def rank0_print(ctx: DistributedContext, message: str) -> None:
    if ctx.is_master:
        print(message, flush=True)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return False


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        scale = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_k.weight, -scale, scale)
            torch.nn.init.uniform_(block.attn.c_v.weight, -scale, scale)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -scale, scale)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -scale, scale)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(char in "SL" for char in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def optimizer_parameter_groups(self):
        matrix_params = []
        for name, param in self.transformer.h.named_parameters():
            if param.ndim != 2:
                raise ValueError(f"Expected transformer block parameter {name} to be 2D for Muon, found shape {tuple(param.shape)}")
            matrix_params.append(param)

        groups = {
            "matrix": matrix_params,
            "embedding": list(self.transformer.wte.parameters()),
            "value_embeds": list(self.value_embeds.parameters()),
            "lm_head": list(self.lm_head.parameters()),
            "resid": [self.resid_lambdas],
            "x0": [self.x0_lambdas],
        }

        seen = set()
        for group_name, params in groups.items():
            for param in params:
                if id(param) in seen:
                    raise ValueError(f"Parameter assigned more than once in optimizer groups: {group_name}")
                seen.add(id(param))

        all_params = list(self.parameters())
        if len(seen) != len(all_params):
            missing = [name for name, param in self.named_parameters() if id(param) not in seen]
            raise ValueError(f"Optimizer grouping missed parameters: {missing}")
        return groups

    def forward(self, idx, targets=None, reduction="mean"):
        _, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)
        return logits


def select_flash_kernel_repo(policy: str, device: torch.device) -> str:
    if policy != "auto":
        return policy
    capability = torch.cuda.get_device_capability(device)
    return DEFAULT_FLASH_KERNEL_REPOS[0] if capability == (9, 0) else DEFAULT_FLASH_KERNEL_REPOS[1]


def load_flash_attention_interface(cache, repo_id: str):
    manifest = load_kernel_manifest(cache)
    entry = manifest.get(repo_id)
    if entry is None:
        raise FileNotFoundError(
            f"Kernel repo '{repo_id}' is missing from {cache.kernel_dir}. Run `python prepare.py --cache-root {cache.cache_root}` on a login node first."
        )
    repo_path = entry.get("path")
    package_name = entry.get("package_name") or package_name_from_repo_id(repo_id)
    if repo_path is None or not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Kernel repo '{repo_id}' is not materialized at {repo_path!r}. Re-run `python prepare.py --cache-root {cache.cache_root}` on a login node."
        )
    return get_local_kernel(Path(repo_path), package_name).flash_attn_interface


def build_model_config(cfg, vocab_size: int) -> GPTConfig:
    base_dim = cfg.depth * cfg.aspect_ratio
    model_dim = ((base_dim + cfg.head_dim - 1) // cfg.head_dim) * cfg.head_dim
    num_heads = model_dim // cfg.head_dim
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=cfg.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=cfg.window_pattern,
    )


def build_optimizers(model: GPT, cfg, ctx: DistributedContext):
    groups = model.optimizer_parameter_groups()
    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    rank0_print(ctx, f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

    adamw_groups = [
        {
            "name": "lm_head",
            "params": groups["lm_head"],
            "lr": cfg.unembedding_lr * dmodel_lr_scale,
            "betas": (cfg.adam_beta1, cfg.adam_beta2),
            "eps": 1e-10,
            "weight_decay": 0.0,
        },
        {
            "name": "embedding",
            "params": groups["embedding"],
            "lr": cfg.embedding_lr * dmodel_lr_scale,
            "betas": (cfg.adam_beta1, cfg.adam_beta2),
            "eps": 1e-10,
            "weight_decay": 0.0,
        },
        {
            "name": "value_embeds",
            "params": groups["value_embeds"],
            "lr": cfg.embedding_lr * dmodel_lr_scale,
            "betas": (cfg.adam_beta1, cfg.adam_beta2),
            "eps": 1e-10,
            "weight_decay": 0.0,
        },
        {
            "name": "resid",
            "params": groups["resid"],
            "lr": cfg.scalar_lr * 0.01,
            "betas": (cfg.adam_beta1, cfg.adam_beta2),
            "eps": 1e-10,
            "weight_decay": 0.0,
        },
        {
            "name": "x0",
            "params": groups["x0"],
            "lr": cfg.scalar_lr,
            "betas": (0.96, 0.95),
            "eps": 1e-10,
            "weight_decay": 0.0,
        },
    ]
    muon_groups = [
        {
            "name": "matrix",
            "params": groups["matrix"],
            "lr": cfg.matrix_lr,
            "weight_decay": cfg.weight_decay,
            "momentum": 0.95,
            "nesterov": True,
            "ns_steps": 5,
            "eps": 1e-10,
        }
    ]

    adamw_optimizer = torch.optim.AdamW(adamw_groups)
    muon_optimizer = torch.optim.Muon(muon_groups)

    for optimizer in (adamw_optimizer, muon_optimizer):
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
    return adamw_optimizer, muon_optimizer


def get_lr_multiplier(progress: float, cfg) -> float:
    if progress < cfg.warmup_ratio:
        return progress / cfg.warmup_ratio if cfg.warmup_ratio > 0 else 1.0
    if progress < 1.0 - cfg.warmdown_ratio:
        return 1.0
    cooldown = (1.0 - progress) / cfg.warmdown_ratio if cfg.warmdown_ratio > 0 else 0.0
    return cooldown + (1 - cooldown) * cfg.final_lr_frac


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress: float, cfg) -> float:
    return cfg.weight_decay * (1 - progress)


def update_optimizer_schedules(adamw_optimizer, muon_optimizer, cfg, progress: float, step: int) -> float:
    lrm = get_lr_multiplier(progress, cfg)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress, cfg)
    for group in adamw_optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    for group in muon_optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        group["momentum"] = muon_momentum
        group["weight_decay"] = muon_weight_decay
    return lrm


def init_distributed_context() -> DistributedContext:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        local_rank = 0

    if local_rank >= torch.cuda.device_count():
        raise SystemExit(f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_distributed=is_distributed,
        is_master=rank == 0,
        device=device,
    )


def cleanup_distributed(ctx: DistributedContext | None) -> None:
    if ctx is not None and ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def maybe_launch_distributed(cfg, argv: list[str]) -> None:
    if cfg.no_spawn:
        return
    if "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return
    if cfg.nproc_per_node <= 1:
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training.")

    visible_gpus = torch.cuda.device_count()
    if visible_gpus < cfg.nproc_per_node:
        raise SystemExit(
            f"Requested {cfg.nproc_per_node} local ranks but only {visible_gpus} GPUs are visible. "
            "Pass --nproc-per-node with a smaller value for debug runs."
        )

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(cfg.nproc_per_node),
        str(Path(__file__).resolve()),
        *argv,
        "--no-spawn",
    ]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


def train_worker(cfg) -> None:
    global fa3

    ctx = init_distributed_context()
    try:
        t_start = time.time()
        torch.manual_seed(cfg.seed + ctx.rank)
        torch.cuda.manual_seed(cfg.seed + ctx.rank)
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats(ctx.device)

        flash_repo = select_flash_kernel_repo(cfg.flash_kernel_repo, ctx.device)
        fa3 = load_flash_attention_interface(cfg.cache, flash_repo)

        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        tokenizer = Tokenizer.from_directory(cfg.cache.tokenizer_dir)
        vocab_size = tokenizer.get_vocab_size()
        rank0_print(ctx, f"Vocab size: {vocab_size:,}")

        config = build_model_config(cfg, vocab_size)
        rank0_print(ctx, f"Model config: {asdict(config)}")

        with torch.device("meta"):
            raw_model = GPT(config)
        raw_model.to_empty(device=ctx.device)
        raw_model.init_weights()

        param_counts = raw_model.num_scaling_params()
        if ctx.is_master:
            print("Parameter counts:")
            for key, value in param_counts.items():
                print(f"  {key:24s}: {value:,}")

        num_params = param_counts["total"]
        num_flops_per_token = raw_model.estimate_flops()
        rank0_print(ctx, f"Estimated FLOPs per token: {num_flops_per_token:e}")

        tokens_per_fwdbwd = ctx.world_size * cfg.device_batch_size * MAX_SEQ_LEN
        if cfg.total_batch_size % tokens_per_fwdbwd != 0:
            raise ValueError(
                f"Invalid batch configuration: total_batch_size={cfg.total_batch_size}, world_size={ctx.world_size}, "
                f"device_batch_size={cfg.device_batch_size}, seq_len={MAX_SEQ_LEN}. "
                f"One synchronized microstep already uses {tokens_per_fwdbwd} tokens, so lower --device-batch-size "
                f"or raise --total-batch-size to a multiple of {tokens_per_fwdbwd}."
            )
        grad_accum_steps = cfg.total_batch_size // tokens_per_fwdbwd

        adamw_optimizer, muon_optimizer = build_optimizers(raw_model, cfg, ctx)

        compiled_model = torch.compile(raw_model, dynamic=False)
        train_model = compiled_model
        if ctx.is_distributed:
            train_model = DDP(compiled_model, device_ids=[ctx.local_rank], broadcast_buffers=False)

        train_loader = make_dataloader(
            tokenizer,
            cfg.device_batch_size,
            MAX_SEQ_LEN,
            "train",
            data_dir=cfg.cache.data_dir,
            rank=ctx.rank,
            world_size=ctx.world_size,
            device=ctx.device,
        )
        x, y, epoch = next(train_loader)

        rank0_print(ctx, f"Time budget: {TIME_BUDGET}s")
        rank0_print(ctx, f"Gradient accumulation steps: {grad_accum_steps}")

        smooth_train_loss = 0.0
        total_training_time = 0.0
        step = 0

        train_model.train()
        raw_model.zero_grad(set_to_none=True)

        while True:
            torch.cuda.synchronize()
            t0 = time.time()
            for micro_step in range(grad_accum_steps):
                sync_ctx = nullcontext()
                if ctx.is_distributed and micro_step < grad_accum_steps - 1:
                    sync_ctx = train_model.no_sync()
                with sync_ctx:
                    with autocast_ctx:
                        loss = train_model(x, y)
                    train_loss = loss.detach()
                    (loss / grad_accum_steps).backward()
                x, y, epoch = next(train_loader)

            progress = min(total_training_time / TIME_BUDGET, 1.0)
            lrm = update_optimizer_schedules(adamw_optimizer, muon_optimizer, cfg, progress, step)
            adamw_optimizer.step()
            muon_optimizer.step()
            raw_model.zero_grad(set_to_none=True)

            train_loss_tensor = train_loss.float().detach()
            if ctx.is_distributed:
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
            train_loss_f = train_loss_tensor.item()

            if math.isnan(train_loss_f) or train_loss_f > 100:
                raise RuntimeError("Training loss exploded or became NaN.")

            torch.cuda.synchronize()
            dt = time.time() - t0
            if step > 10:
                total_training_time += dt

            ema_beta = 0.9
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
            pct_done = 100 * progress
            tok_per_sec = int(cfg.total_batch_size / dt)
            mfu = 100 * num_flops_per_token * cfg.total_batch_size / dt / H100_BF16_PEAK_FLOPS
            remaining = max(0.0, TIME_BUDGET - total_training_time)

            if ctx.is_master:
                print(
                    f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
                    f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
                    f"mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ",
                    end="",
                    flush=True,
                )

            if step == 0:
                gc.collect()
                gc.freeze()
                gc.disable()
            elif (step + 1) % 5000 == 0:
                gc.collect()

            step += 1
            should_stop = step > 10 and total_training_time >= TIME_BUDGET
            if ctx.is_distributed:
                stop_tensor = torch.tensor([int(should_stop)], dtype=torch.int32, device=ctx.device)
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                should_stop = bool(stop_tensor.item())
            if should_stop:
                break

        if ctx.is_master:
            print()

        total_tokens = step * cfg.total_batch_size
        eval_model = raw_model
        eval_model.eval()

        val_bpb_tensor = torch.zeros(1, dtype=torch.float64, device=ctx.device)
        if ctx.is_master:
            print("Starting final evaluation...", flush=True)
            with autocast_ctx:
                val_bpb_tensor[0] = evaluate_bpb(
                    eval_model,
                    tokenizer,
                    cfg.device_batch_size,
                    data_dir=cfg.cache.data_dir,
                    tokenizer_dir=cfg.cache.tokenizer_dir,
                    device=ctx.device,
                )
            print("Final evaluation complete.", flush=True)
        if ctx.is_distributed:
            dist.broadcast(val_bpb_tensor, src=0)

        peak_vram_tensor = torch.tensor(
            [torch.cuda.max_memory_allocated(device=ctx.device) / 1024 / 1024],
            dtype=torch.float64,
            device=ctx.device,
        )
        if ctx.is_distributed:
            dist.all_reduce(peak_vram_tensor, op=dist.ReduceOp.MAX)

        t_end = time.time()
        steady_state_mfu = 0.0
        if total_training_time > 0 and step > 10:
            steady_state_mfu = 100 * num_flops_per_token * cfg.total_batch_size * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS

        if ctx.is_master:
            print("---")
            print(f"val_bpb:          {val_bpb_tensor.item():.6f}")
            print(f"training_seconds: {total_training_time:.1f}")
            print(f"total_seconds:    {t_end - t_start:.1f}")
            print(f"peak_vram_mb:     {peak_vram_tensor.item():.1f}")
            print(f"mfu_percent:      {steady_state_mfu:.2f}")
            print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
            print(f"num_steps:        {step}")
            print(f"num_params_M:     {num_params / 1e6:.1f}")
            print(f"depth:            {cfg.depth}")
    finally:
        cleanup_distributed(ctx)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = parse_train_config(argv)
    maybe_launch_distributed(cfg, argv)
    train_worker(cfg)


if __name__ == "__main__":
    main()
