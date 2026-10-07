# ReproBench

A small benchmark for the thing ReproFix does: take a repository that is broken in some way, and get it to reproduce
its documented result.

**No result from a real model is committed to this repository.** The only committed run
(`benchmark/results/oracle-router.{json,md}`) was made with a scripted "oracle" that replays each task's reference fix.
It exists to prove the harness works and says nothing about Nemotron or any other model. See
[Running it for real](#running-it-for-real).

## Design

Every task starts from the same small, working project: a multi-layer perceptron trained on a synthetic
image-like dataset (10 classes, 3x8x8 inputs), whose README documents `val_accuracy: 0.881` (0.939 for the PyTorch
task, see below). One realistic fault is injected (the demo task injects three). Because the data is generated, 29 of the
30 tasks need nothing but numpy and pytest: no dataset download, no GPU, seconds per run, so a benchmark run is cheap and
reproducible anywhere. Of those 29, 25 run entirely offline in the test suite (against the host's numpy and pytest), and four
need PyPI for their `pip install` (the three dependency tasks and the demo). The 30th task is real PyTorch and is the
exception (see [The PyTorch task](#the-pytorch-task)).

The trade-off is realism. These are not real research repositories, and they are far easier to read and reason about
than one. Treat the benchmark as a regression and ablation tool for ReproFix, not as a measure of how it would do on
arbitrary code.

Tasks are written by `benchmark/build_tasks.py`. Each task directory holds `task.json` (the spec, ground truth and
reference fix as exact search/replace edits) and `repo/` (the broken repository). The generator **only adds tasks that
are missing**: it first checks that every committed task is exactly what the generator would produce (byte for byte).
If any differs it prints which one and writes nothing, so results published against a task stay valid; only when all
match does it write the tasks that are missing. `--check` does only the comparison (it exits non-zero if a task differs
or is missing; the test suite calls the same comparison, `differences()`), `--measure` runs each broken state and prints
what it does (the PyTorch task is skipped unless you pass `--torch-deps` with a directory that already has torch
installed), and `--rebuild` regenerates everything and therefore invalidates old results.

Tasks 21 to 30 were added after the first release; tasks 1 to 20 are byte-identical to it (compared against the files in
the original archive). Each new fault was **measured** before it was kept. Some candidates were dropped because they did
not break this dataset: for example, deleting the division by batch size from the gradient still scores 0.8825 here (the
larger effective learning rate happens to help), and removing the ReLU mask scores 0.903, higher than the working code.
Those two figures come from my measurement runs while building the tasks and are not recorded anywhere in the repository;
`--measure` reproduces the numbers for the tasks that are kept.

## The 30 tasks

| task | category | difficulty | first symptom | what is wrong |
|---|---|---|---|---|
| `arch_classifier_dim` | architecture | easy | crash | Output layer input size does not match the hidden layer |
| `arch_num_classes` | architecture | easy | crash | num_classes in config smaller than the number of classes in the data |
| `cfg_epochs` | configuration | medium | wrong result | epochs set to 1 although the README documents 30 |
| `cfg_key_mismatch` | configuration | easy | crash | config.json uses 'learning_rate' but train.py reads 'lr' |
| `cfg_wrong_dtype` | configuration | easy | crash | Hidden size stored as a string in config.json |
| `ckpt_hidden_mismatch` | checkpoint | medium | crash | config hidden size differs from the released checkpoint |
| `ckpt_key_names` | checkpoint | medium | crash | Checkpoint keys are lowercase but the loader asks for uppercase |
| `ckpt_wrong_path` | checkpoint | easy | crash | Evaluation script loads the checkpoint from a directory that does not exist |
| `code_none_shuffle` | code | easy | crash | rng.shuffle() result used as an index array |
| `code_off_by_one` | code | easy | crash | Off-by-one when printing the last loss |
| `data_label_offset` | data | medium | wrong result | Training labels shifted by one; validation labels are not |
| `data_label_shuffle` | data | medium | wrong result | Training labels shuffled independently of the images |
| `data_val_labels` | data | medium | wrong result | Validation labels drawn from a second, unrelated sample |
| `demo_broken_image_classifier` | multi | hard | install failure | Demo: dependency conflict + classifier dimension + wrong normalisation |
| `dep_conflicting_pins` | dependency | easy | install failure | Contradictory numpy pins in requirements.txt |
| `dep_missing_requirement` | dependency | medium | crash | train.py imports tqdm but requirements.txt omits it |
| `dep_nonexistent_version` | dependency | easy | install failure | Pinned numpy version that does not exist |
| `device_cuda_config` | device | easy | crash | Simulated device mismatch: config asks for an accelerator the machine does not have |
| `eval_floor_division` | evaluation | easy | wrong result | Accuracy computed with floor division |
| `eval_inverted_metric` | evaluation | easy | wrong result | Error rate reported as accuracy |
| `eval_on_train_split` | evaluation | medium | wrong result | 'Validation' accuracy computed on the training split |
| `pre_double_scaling` | preprocessing | medium | wrong result | Pixels rescaled to [0, 1] and then standardised with [0, 255] statistics |
| `pre_val_not_normalized` | preprocessing | medium | wrong result | Validation split is not normalised |
| `pre_wrong_normalization` | preprocessing | medium | wrong result | Generic 0.5/0.5 normalisation instead of the dataset statistics |
| `torch_double_softmax` | training | medium | wrong result | PyTorch: model returns probabilities but the loss expects logits |
| `train_init_scale` | training | medium | wrong result | He initialisation formula inverted for the first layer |
| `train_lr_schedule` | training | medium | wrong result | Learning-rate decay factor 0.5 instead of the documented 0.95 |
| `train_lr_too_high` | training | easy | wrong result | Learning rate 1000x the documented value |
| `train_model_reinit` | training | medium | wrong result | Model re-initialised at the start of every epoch |
| `train_update_sign` | training | hard | wrong result | Gradient ascent on the output weights |

Compared with the categories in the original brief (dependency, architecture, data, preprocessing, training,
evaluation, GPU, configuration, checkpoint, code): every category has tasks, but **the GPU category is only
simulated**. `device_cuda_config` is pure numpy: `device.py` stands in for an accelerator check and `config.json` asks for
`"cuda"` on a machine that only has `"cpu"`. Its `task.json` says so (`notes`). It tests how the agent handles a
config/hardware mismatch, not real GPU behaviour. There is no GPU anywhere in this benchmark. "Training: learning-rate
bug" is covered by `train_lr_too_high` (the weights overflow to NaN) and `train_lr_schedule`.

### The PyTorch task

`torch_double_softmax` is a real PyTorch project (same data, preprocessing and metric modules; a `torch` MLP with
dropout and a `torch` training loop). `MLP.forward` applies softmax and `CrossEntropyLoss` then applies log-softmax again, a
classic mistake. Measured on the machine this was built on (Python 3.13, 2 vCPUs, torch 2.14.1 from PyPI, CPU only):

| | |
|---|---|
| documented `val_accuracy` | 0.939 (the working template) |
| broken repository | 0.4005 |
| one training run | about 3.5 s on one thread |
| `pip install torch` through the sandbox | the PyPI Linux wheel brings the CUDA runtime libraries: about 2.9 GB to download, 5.3 GB installed |
| validation (broken run, fix, hidden check; two installs) | 229 s in total |

Consequences you should know about:

- It is the only task that needs the network at install time for anything large, and it needs about 5.3 GB of disk per workspace.
  Its `task.json` sets `timeout_s: 1200`; the harness passes that to every command and to the run's budgets (every other
  task keeps the 300 s default).
- The documented 0.939 and the broken 0.4005 were measured with that one torch version. `requirements.txt` says
  `torch>=2.2`, so another version may move the working result by a little. The broken result is far outside the
  tolerance of 0.03 (and the hidden check's 0.06), so the task still discriminates; I have not run any other torch version.
- The default (offline) test suite skips it. `REPROFIX_TEST_NETWORK=1 REPROFIX_TEST_TORCH=1 pytest tests/test_benchmark.py -k pytorch`
  validates it against real pip.
- A CPU-only wheel from `download.pytorch.org` would be much smaller. That host was not reachable from the build
  environment, so it is untested; adding the index to the task's requirements is the obvious experiment.

## What is measured

Scoring is deterministic: no model judges another model.

| Stage | Passes when |
|---|---|
| understood | The run command ReproFix inferred equals the task's command. The command is **not** given to it. |
| reproduced | The baseline run failed in the way the task documents (install failure, crash, or wrong metric), and, where the task specifies one, the error text matches its pattern. |
| root cause | The *confirmed* hypotheses together name every ground-truth file and (for single-fault tasks) the ground-truth category. **This is a proxy**: it checks where the model said the problem was, not whether its explanation was right. |
| patch generated | At least one file was changed and kept. |
| patch verified | ReproFix's own verification passed **and** the hidden check passed. |

**Hidden check** (never shown to the agent), run on the agent's final workspace:

1. Protected files (`tests/**`, `test_*.py`, `README*`, `LICENSE*`) are byte-identical to the originals.
2. A fresh run with a different data seed (`--seed 1234`) lands within a wider tolerance (0.06) of the documented value.
   This catches a "fix" that only works for the seed the agent saw.
3. The printed metric equals an independent recomputation from `artifacts/predictions.npz` (within 0.001). This catches
   a program that prints a number it did not compute.

The summary also records tokens, model calls, model latency, wall time, mean experiments per task, and cost (only
for tiers that have a configured price, and labelled so when some do not).

## Checking the tasks themselves

```bash
reprofix bench list
reprofix bench validate          # needs a sandbox; add REPROFIX_SANDBOX=local REPROFIX_ALLOW_UNSAFE_LOCAL=1 on a dev machine
```

`validate` involves no model. For each task it shows that the broken repository fails with the documented symptom,
applies the reference fix, and shows that the fixed repository verifies and passes the hidden check. All 30 tasks
passed this (`30/30 tasks valid`), using real `pip` for the three dependency tasks, the demo task and the PyTorch task. The
25 offline tasks are also validated, without a network, by the test suite, which additionally asserts that the generator
reproduces every committed task byte for byte.

## Running it for real

```bash
cp .env.example .env                       # add NEBIUS_API_KEY
reprofix doctor --live                     # confirm the three model IDs and the base URL work with your key

reprofix bench run --llm nebius --router-mode router
reprofix bench run --llm nebius --router-mode super-only
reprofix bench run --llm nebius --router-mode ultra-only

reprofix bench compare benchmark/results/nebius-router.json \
                       benchmark/results/nebius-super-only.json \
                       benchmark/results/nebius-ultra-only.json
```

Results are written to `benchmark/results/<label>.{json,md}`. A run restricted with `--only` is labelled
`...-subset` so it cannot be mistaken for, or overwrite, a full run.

If you report numbers, report what you measured. Some things to keep honest:

- **One run of 30 tasks is a small sample.** Models are not deterministic (temperature 0.2). A difference of one or two
  tasks between routing modes is within noise; repeat each mode several times before claiming an ordering.
- **Cost depends on the prices in your `.env`** (defaults are Nebius's catalog prices on 2026-10-02) and on which
  tiers the router actually used. The per-tier call counts are in the JSON.
- **Say what the router is compared with.** The point of `super-only` and `ultra-only` is to test whether routing
  (Nano for summaries, Super for ordinary diagnosis and repair, Ultra for wrong-result diagnosis and after rejections)
  buys anything over a single model. That has not been measured.

## The oracle run

```bash
reprofix bench run --llm oracle
```

A scripted backend answers every agent call from the task's own reference fix. It scores 30/30 on every stage **by
construction** (the committed run took 306 s, 88 s of it the PyTorch task). It is useful for one thing: if the oracle
fails a task, something in the harness, sandbox, patching or verification is broken. Its results are labelled
`scripted` and "NOT Nemotron" in the JSON, the markdown, the CLI, and the UI.

It did its job while tasks 21 to 30 were added. The first full run scored 29/30: the oracle replayed only the first of the
PyTorch fix's two edits, which left the repository half fixed, and the "device" task could not earn the root-cause stage
because the diagnoser's category list had no `device` entry. The oracle now replays a whole single-fault fix, the
category exists, and both have regression tests. Neither defect was in the tasks themselves.

## Limitations

- Synthetic, CPU-only, single-fault (one three-fault demo) tasks, all on the same small MLP-on-synthetic-data project;
  one of them uses PyTorch, the rest numpy. No GPU tasks (the "device" task is a simulation), no real-world repositories,
  no multi-file refactors, no flaky environments.
- The `root cause` stage is a file-and-category proxy.
- Faults were written by the same people who wrote the agent; they reflect common failure classes, not an independent sample.
- A task counts as `understood` only if the inferred command matches exactly; an equivalent command would be scored as a miss.
- Wall-clock and latency figures depend on the machine, the network, and Nebius load at the time.
