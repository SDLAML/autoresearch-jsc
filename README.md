# autoresearch

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

This repository is a small, cluster-oriented autoresearch setup: an agent modifies the training code, submits short runs, reads the result, and iterates.

The current version of this repo is designed for a JSC-style workflow:
- login node for environment setup and `prepare.py`
- single-node 4-GPU training jobs by default
- Slurm submission through `submit_template.sbatch`
- shared repo-local cache under `./.cache/autoresearch/`

## Core files

- `prepare.py`: fixed data preparation, tokenizer loading, dataloading, and evaluation harness
- `train.py`: training entrypoint, default single-node DDP launcher, model, and optimizer setup
- `runtime_config.py`: train-time runtime defaults and CLI surface
- `program.md`: operator and agent instructions for the experiment loop
- `submit_template.sbatch`: default Slurm submission template

## Environment

The Python environment lives in `llm_env`.

First-time setup:
```bash
source llm_env/setup.sh
```

For every shell where you want to use the repo:
```bash
source llm_env/activate.sh
```

Compute nodes do not have internet access. Install packages and run `prepare.py` on a login node first.

## Prepare on a login node

```bash
source llm_env/activate.sh
python prepare.py
```

This populates `./.cache/autoresearch/` with:
- parquet shards
- tokenizer files
- offline kernel assets needed by `train.py`

## Run training

Local debug run on one GPU:
```bash
source llm_env/activate.sh
python train.py --nproc-per-node 1
```

Default 4-GPU single-node run:
```bash
source llm_env/activate.sh
python train.py
```

The default path auto-launches single-node DDP with 4 ranks.

## Submit a Slurm job

Default submission:
```bash
sbatch submit_template.sbatch
```

Pass extra `train.py` arguments through `sbatch`:
```bash
sbatch submit_template.sbatch --device-batch-size 48
```

Logs are written to:
```text
./.cache/slurm_output/<jobid>.log
```

The sbatch template checks that the prepared cache already exists and fails fast if data, tokenizer files, or kernel assets are missing.

## Metrics and outputs

Training uses a fixed 5-minute measured training budget.

At the end of a run, `train.py` prints a summary including:
- `val_bpb`
- `training_seconds`
- `peak_vram_mb`
- `mfu_percent`
- `num_steps`
- `num_params_M`

Example extraction from a Slurm log:
```bash
grep "^val_bpb:\|^peak_vram_mb:" ./.cache/slurm_output/<jobid>.log
```

`train.py` does not write `results.tsv` automatically. Results logging is a separate workflow described in `program.md`.

## What is fixed

Treat these as infrastructure, not research surface:
- `prepare.py`
- the evaluation harness in `prepare.py`
- the fixed time budget and other fixed evaluation-side constraints

The intended research surface is `train.py`, and when needed, `runtime_config.py`.

## More detail

For the actual experiment loop, logging rules, keep/discard criteria, and agent behavior, read:
- `program.md`
