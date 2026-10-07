# ReproBench results (oracle-router)

> **LLM backend: `scripted` -- NOT Nemotron.** These numbers validate the pipeline and benchmark only; they say nothing about model capability.

- date: 2026-10-03 13:10
- router mode: router
- sandbox: local
- models: n/a (scripted)
- tasks: 30

| stage | passed |
|---|---|
| understood | 30/30 |
| reproduced | 30/30 |
| root cause | 30/30 |
| patch generated | 30/30 |
| patch verified | 30/30 |

tokens: 0 · LLM calls: 154 · LLM latency: 0.0s · wall: 305.9s · mean experiments/task: 1.07 · cost: n/a

| task | category | status | understood | reproduced | root cause | patch generated | patch verified | experiments |
|---|---|---|---|---|---|---|---|---|
| arch_classifier_dim | architecture | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| arch_num_classes | architecture | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| cfg_epochs | configuration | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| cfg_key_mismatch | configuration | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| cfg_wrong_dtype | configuration | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| ckpt_hidden_mismatch | checkpoint | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| ckpt_key_names | checkpoint | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| ckpt_wrong_path | checkpoint | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| code_none_shuffle | code | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| code_off_by_one | code | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| data_label_offset | data | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| data_label_shuffle | data | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| data_val_labels | data | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| demo_broken_image_classifier | multi | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 3 |
| dep_conflicting_pins | dependency | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| dep_missing_requirement | dependency | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| dep_nonexistent_version | dependency | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| device_cuda_config | device | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| eval_floor_division | evaluation | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| eval_inverted_metric | evaluation | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| eval_on_train_split | evaluation | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| pre_double_scaling | preprocessing | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| pre_val_not_normalized | preprocessing | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| pre_wrong_normalization | preprocessing | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| torch_double_softmax | training | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| train_init_scale | training | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| train_lr_schedule | training | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| train_lr_too_high | training | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| train_model_reinit | training | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
| train_update_sign | training | verified | ✓ | ✓ | ✓ | ✓ | ✓ | 1 |
