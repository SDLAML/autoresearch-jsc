# autoresearch

This repository is an experiment in having the LLM do its own research.

## Environment

The Python environment lives in `llm_env`.

First-time setup:
`source llm_env/setup.sh`

For every shell where you want to use the project:
`source llm_env/activate.sh`

Compute nodes do not have internet access. Any package installation, data downloads, tokenizer building, or kernel asset downloads must be done on a login node first.

## Prepare on a login node

Run:
`source llm_env/activate.sh`
`python prepare.py`

This populates the shared cache under `./.cache/autoresearch/` with:
- parquet shards
- tokenizer files
- offline kernel assets required by `train.py`

Do not run `prepare.py` on a compute node.

## Experiment setup

To set up a new experiment, work with the user to:

1. Agree on a run tag, usually based on the date, for example `mar5`.
2. Create a fresh branch: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files for context:
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `runtime_config.py`
4. Verify the prepared cache exists under `./.cache/autoresearch/`. If not, tell the human to run `python prepare.py` on a login node.
5. Initialize `results.tsv` with only the header row if it does not already exist.
6. Confirm the setup and start experimenting.

## What you may change

You may change:
- `train.py`
- `runtime_config.py`

You may not change:
- `prepare.py`
- the evaluation harness in `prepare.py`
- project dependencies

`prepare.py` is treated as fixed infrastructure. It owns data preparation, tokenizer loading, dataloading, and evaluation.

## Training model

Each experiment runs on one node with 4 A100 40GB GPUs by default.

The training time budget is fixed at 5 minutes of measured training time, excluding startup and compilation overhead. The Slurm job walltime is set higher to leave room for environment setup, process launch, and final evaluation.

The goal is simple: minimize `val_bpb`.

Everything in `train.py` and `runtime_config.py` is fair game if it improves the metric without making the code unreasonably ugly or fragile.

## Simplicity rule

All else being equal, simpler is better.

A tiny gain that adds ugly complexity is usually not worth keeping. Removing code while keeping the same result, or improving the result, is a win.

## First run

The first run on a new branch should establish a baseline using the current default training setup.

## Output format

At the end of training, the script prints a summary like this:

```text
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

To extract the main metrics from a log file:

```bash
grep "^val_bpb:\|^peak_vram_mb:" ./.cache/slurm_output/<jobid>.log
```

## Logging results

When an experiment finishes, record it in `results.tsv`.

Use tab-separated format with this header:

```text
commit	val_bpb	memory_gb	status	description
```

Columns:
1. short git commit hash
2. `val_bpb`, or `0.000000` for crashes
3. peak memory in GB, rounded to one decimal place, or `0.0` for crashes
4. one of `keep`, `discard`, or `crash`
5. short description of the experiment

Example:

```text
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

Do not commit `results.tsv`.

## Experiment loop

The experiment runs on a dedicated branch such as `autoresearch/mar5`.

Loop:

1. Inspect the current branch and commit.
2. Make an experimental change in `train.py` or `runtime_config.py`.
3. Commit the change.
4. Submit one or more jobs for that commit:
   `sbatch submit_template.sbatch [extra train.py args]`
5. Read the result from the Slurm log:
   `grep "^val_bpb:\|^peak_vram_mb:" ./.cache/slurm_output/<jobid>.log`
6. If the run crashed, inspect the tail of the log:
   `tail -n 50 ./.cache/slurm_output/<jobid>.log`
7. Record the outcome in `results.tsv`.
8. If all submitted jobs for the commit are finished, decide whether to keep the commit or reset it.
9. Keep the commit only if the result is better or otherwise clearly worth the added complexity.

Running multiple jobs for the same commit is acceptable for controlled sweeps such as learning rate or batch size, but do not do this casually. Each kept commit should remain self-contained and reproducible.

## Crashes and timeouts

A run that crashes because of a trivial bug should usually be fixed and rerun.

A run that fails because the idea is fundamentally bad should be marked as `crash` or `discard`, logged, and abandoned.

The Slurm template uses a 10-minute walltime. That is intentional. It leaves room for startup and final evaluation and should not be shortened casually.

## Submit a training job

The project includes `submit_template.sbatch`.

Default submission:
`sbatch submit_template.sbatch`

Submit with extra `train.py` arguments:
`sbatch submit_template.sbatch --device-batch-size 48`

Logs are written to:
`./.cache/slurm_output/<jobid>.log`

The template checks that the prepared cache already exists and fails fast if data, tokenizer files, or kernel assets are missing.

There are two relevant queues on the cluster: `develbooster` and `booster`.

The template currently targets `develbooster`. That is appropriate for short development runs. If needed, you may switch to `booster`, but do so deliberately.

## Autonomy rule

Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
