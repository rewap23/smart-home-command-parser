# Smart Home Command Parser — Model Card

## Model summary

- **Model name:** smart-home-command-parser-transformer
- **Version:** 0.1.0
- **Framework:** PyTorch
- **Architecture:** Encoder-only Transformer with six classification heads
- **Task:** Convert natural-language smart-home commands into structured fields:
  `intent`, `action`, `device`, `location`, `value`, and `unit`
- **Training hardware:** Google Colab CUDA GPU
- **Checkpoint epoch:** 20

## Supported command examples

| Input command | Expected parse |
|---|---|
| Turn on the kitchen lights | `device_control`, `turn_on`, `light`, `kitchen` |
| Set the garage heater to 62 degrees | `device_control`, `set_temperature`, `thermostat`, `garage`, `62`, `fahrenheit` |
| Set the office speaker volume to 30 percent | `device_control`, `set_volume`, `speaker`, `office`, `30`, `percent` |
| Start a timer for ten minutes | `unsupported`, `none`, `none`, `none`, `none`, `none` |

## Dataset

- **Total examples:** 29,448
- **Unsupported examples:** 1,479
- **Unsupported template IDs:** 51
- **Data source:** Programmatically generated synthetic smart-home commands
- **Split method:** Template-family split, so a `template_id` occurs in only one of train, validation, or test.

## Held-out test results

| Metric | Value |
|---|---:|
| Test examples | 4,357 |
| Test loss | 0.0008974845 |
| Intent accuracy | 1.0000 |
| Action accuracy | 1.0000 |
| Device accuracy | 1.0000 |
| Location accuracy | 1.0000 |
| Value accuracy | 1.0000 |
| Unit accuracy | 1.0000 |
| Exact-match accuracy | 1.0000 |

## Interpretation

The model achieved 100% exact-match accuracy on the synthetic held-out test set. This demonstrates that the model can learn the generated task distribution, including the unsupported intent. It does not establish equivalent performance on open-ended real-world language.

## Limitations

- Training and evaluation data are synthetic and template-derived.
- The held-out test set may remain structurally similar to training examples.
- The parser has not been evaluated on speech-to-text errors, multilingual commands, colloquialisms, typos, ambiguous instructions, or unseen device names.
- Confidence calibration has not yet been validated.
- The model must not directly control physical devices.
- High-risk commands such as unlocking doors or opening garage doors require separate authorization and explicit confirmation logic.

## Safety policy

The model output is only a proposed structured parse. The serving application must:

1. Validate the field combination against the command ontology.
2. Reject unsupported or low-confidence commands.
3. Route high-risk actions through an explicit confirmation flow.
4. Simulate execution until an authorization-aware device integration exists.