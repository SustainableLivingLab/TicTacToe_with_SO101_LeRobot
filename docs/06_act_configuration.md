# 06. ACT Model Configuration

## Which ACT variant this project uses

LeRobot's policy factory (`TicTacToe_with_SO101/src/lerobot/policies/factory.py`)
registers three ACT variants:

- `act`: stock LeRobot ACT (`configuration_act.py` / `modeling_act.py`).
- `act_lang`: this project's task-instruction-conditioned ACT
  (`configuration_act_lang.py` / `modeling_act_lang.py`).
- `act_lang_ni`: a further variant (`configuration_act_ni.py` /
  `modeling_act_ni.py`), not used by this project.

Train and run with `--policy.type=act_lang`. All configuration below refers
to `ACTLangConfig` in
`TicTacToe_with_SO101/src/lerobot/policies/act/configuration_act_lang.py`.

## Setting config values

Any field on `ACTLangConfig` can be set from the command line when training:

```bash
python -m lerobot.scripts.train \
    --policy.type=act_lang \
    --policy.dim_model=512 \
    --policy.n_encoder_layers=4 \
    --policy.n_decoder_layers=1 \
    ...
```

Or in Python, by constructing `ACTLangConfig(...)` directly and passing it to
`make_policy()`.

## Transformer layer count and width

These are the fields that control model size and capacity. All live in
`configuration_act_lang.py`, lines 106-125.

| Field | Default | What it controls |
|---|---|---|
| `dim_model` | 512 | Hidden dimension of the transformer (encoder, decoder, and the task-embedding buffer, see below). |
| `n_heads` | 8 | Attention heads per transformer layer. |
| `dim_feedforward` | 3200 | Width of the feed-forward sublayer inside each transformer layer. |
| `feedforward_activation` | `"relu"` | Activation function in the feed-forward sublayer. |
| `n_encoder_layers` | 4 | Number of layers in the main transformer encoder (`ACTEncoder`, `is_vae_encoder=False`). |
| `n_decoder_layers` | 1 | Number of layers in the transformer decoder (`ACTDecoder`). Upstream note: the original ACT paper uses 7, but a bug in the reference implementation means only the first layer is actually used, so this codebase matches that behavior with 1. |
| `n_vae_encoder_layers` | 4 | Number of layers in the separate VAE encoder (`ACTEncoder`, `is_vae_encoder=True`), only instantiated when `use_vae=True`. |
| `pre_norm` | `False` | Whether transformer blocks apply LayerNorm before (`True`) or after (`False`) each sublayer. |
| `dropout` | 0.1 | Dropout probability inside transformer layers. |

These counts are consumed directly in `modeling_act_lang.py`:

```python
# ACTEncoder.__init__ (line ~600)
num_layers = config.n_vae_encoder_layers if is_vae_encoder else config.n_encoder_layers
self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])

# ACTDecoder.__init__ (line ~656)
self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
```

To change how many transformer layers the model has, change
`n_encoder_layers`, `n_decoder_layers`, or `n_vae_encoder_layers` in the
config (or pass them as `--policy.n_encoder_layers=N` on the training
command). No code changes needed for layer count; the `nn.ModuleList`
construction already reads from config.

## Vision backbone

| Field | Default | Notes |
|---|---|---|
| `vision_backbone` | `"resnet18"` | Must be a torchvision ResNet variant name (validated in `__post_init__`). |
| `pretrained_backbone_weights` | `"ResNet18_Weights.IMAGENET1K_V1"` | Set to `None` to train the backbone from scratch. |
| `replace_final_stride_with_dilation` | `False` | Swaps the ResNet's final 2x2 stride for a dilated convolution. |

Backbone instantiation is in `ACT.__init__` (`modeling_act_lang.py`, around
line 348), using `torchvision.models` by name, so any backbone name valid
for `getattr(torchvision.models, name)` that is also a ResNet family model
will work.

## Action chunking

| Field | Default | Notes |
|---|---|---|
| `chunk_size` | 100 | Number of future action steps predicted per forward pass. |
| `n_action_steps` | 100 | Number of predicted steps actually executed before re-querying the policy. Must be <= `chunk_size`. |
| `n_obs_steps` | 1 | Fixed at 1; the config raises an error if set otherwise (multi-step observation history is not implemented in this codebase). |
| `temporal_ensemble_coeff` | `None` | If set, enables temporal ensembling across overlapping chunks. Requires `n_action_steps=1`. |

## VAE objective

| Field | Default | Notes |
|---|---|---|
| `use_vae` | `True` | Whether training uses the variational objective (adds a VAE encoder branch). |
| `latent_dim` | 32 | Dimensionality of the VAE latent. At inference, the latent is fixed to all zeros regardless of this setting (see `modeling_act_lang.py`, the `else` branch around line 527). |
| `kl_weight` | 10.0 | Weight of the KL-divergence term when `use_vae=True`. Total loss is `reconstruction_loss + kl_weight * kld_loss`. |

## Optimizer preset

| Field | Default |
|---|---|
| `optimizer_lr` | 1e-5 |
| `optimizer_weight_decay` | 1e-4 |
| `optimizer_lr_backbone` | 1e-5 |

Consumed by `get_optimizer_preset()`, which builds an `AdamWConfig`. Override
with `--policy.optimizer_lr=...` etc. on the training command, same as any
other field.

## The task-instruction conditioning mechanism (project-specific)

This is the modification this project made on top of stock ACT, marked with
`(Ad):` comments in `modeling_act_lang.py`. It is not a learned embedding
table; it is a fixed, hand-constructed one-hot-style buffer, built once at
model construction and never updated by gradient descent.

`ACT.create_task_embeddings()` (`modeling_act_lang.py`, line ~403):

```python
def create_task_embeddings(self):
    """Create 9 fixed embeddings for positions 1-9"""
    embeddings = torch.zeros(9, self.config.dim_model)
    chunk_size = self.config.dim_model // 9

    for i in range(9):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, self.config.dim_model)
        embeddings[i, start_idx:end_idx] = 1.0

    return embeddings
```

This splits `dim_model` into 9 contiguous chunks (one per grid cell) and sets
each chunk to `1.0` for its own position, `0.0` everywhere else. Registered
as a buffer (`self.task_embeddings`), not a parameter, so it is fixed for
the life of the model and included in checkpoints but never trained.

`ACT.prepare_language()` (`modeling_act_lang.py`, line ~422) parses the task
string from the batch (e.g. `"Place at Position 5"`) by reading its last
character, converting to a 0-indexed grid position, and looking up the
corresponding row of `task_embeddings`:

```python
grid_pos = task.lower().strip()[-1]  # last character of the instruction string
if grid_pos not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    raise ValueError(f"Grid position {grid_pos} not in valid range 1-9")
grid_indices = torch.tensor([int(grid_pos) - 1 for ...], device=device)
lang_token = self.task_embeddings[grid_indices]
```

Constraint: this only works for single-digit positions, since it reads only
the last character of the string. Fine for a 3x3 board (positions 1-9), but
would break if the grid ever grew past 9 cells without changing this parsing
logic.

The resulting `task_embed` is injected as an extra token in both the VAE
encoder input sequence and the main transformer encoder input sequence
(`forward()`, lines ~489 and ~540). This is why `num_input_token_encoder`
and `n_1d_tokens` in `__init__` both add `+= 1` with an `(Ad): task token`
comment, one extra position-embedding slot to account for the new token.

To change how many distinct tasks the model conditions on (e.g. a larger
board), both `create_task_embeddings()`'s hardcoded `9` and
`prepare_language()`'s hardcoded `["1", ..., "9"]` range and single-character
parsing would need to change together.

## Vision is architecturally live; low attention is a learned, not structural, property

Interpretability analysis found the trained policy attends almost entirely
to task instruction and joint state, and barely to vision (see
`03_dataset_and_training.md`). This does not mean ACT is structurally blind
to images. The vision pathway is fully wired into the forward pass:

```python
# ACT.__init__
self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)
self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

# ACT.forward
for img in batch["observation.images"]:
    cam_features = self.backbone(img)["feature_map"]
    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)
    cam_features = self.encoder_img_feat_input_proj(cam_features)
    encoder_in_tokens.extend(list(cam_features))
```

Every training and inference image is run through the ResNet backbone,
projected into transformer tokens, and fed into the same encoder attention
mechanism as the task and joint-state tokens. Nothing in the architecture
prevents the model from attending to those tokens.

Attention weight is learned, not fixed. Given this project's dataset (fixed
camera, fixed board, fixed pickup spot, and a task token that already
losslessly encodes the target cell, see `03_dataset_and_training.md`), the
model found it could minimize training loss without relying on vision,
since the task token alone was sufficient. There was no gradient pressure
to route attention there.

Consequence: switching to a different policy architecture (Diffusion
Policy, a VLA such as SmolVLA, or anything else) trained on this same
dataset would very likely reproduce the same near-zero vision attention,
since the shortcut being learned is a property of the data, not of ACT
specifically. Fixing this requires changing what the data makes necessary
to attend to (see `03_dataset_and_training.md`, "Known limitation: this
dataset is not environment-agnostic"), not changing the model. A stronger
pretrained vision backbone or a VLA's pretrained visual encoder could still
help with cross-environment robustness once the data actually requires
vision, but is not a fix for this specific attention pattern on its own.
