# LPI-KAN Core Implementation

This package provides a compact reference implementation of the lightweight
physics-informed KAN framework described in the manuscript. It documents the
main model modules, training framework, and data partitioning protocol.

Raw vehicle data, vehicle identifiers, trained checkpoints, local file paths,
and study-specific visualization scripts are not included.

## Package contents

```text
LPI_KAN_review_package/
|-- README.md
`-- lpi_kan/
    |-- __init__.py
    |-- data_protocol.py
    |-- model.py
    |-- trainer.py
    `-- transfer.py
```

- `model.py` contains the B-spline KAN layers and the physics-informed SOC
  prediction model.
- `trainer.py` contains the main training, validation, model-selection, and
  checkpoint-saving procedures.
- `data_protocol.py` contains the file-level split and sliding-window
  construction functions.
- `transfer.py` contains the target-vehicle fine-tuning strategies.

## Framework overview

### Input and prediction module

Each model sample consists of a 30-step historical input sequence and the
following 30-step SOC target sequence. The input contains vehicle, battery,
environmental, and derived features. The model directly predicts all 30 future
SOC values in one forward pass rather than using autoregressive prediction.

### KAN module

The KAN module provides nonlinear feature representation through learnable
B-spline transformations and maps the historical input sequence to the future
SOC trajectory.

The reported KAN configuration was as follows:

- Input features: 12;
- Input window length: 30 steps;
- Flattened input dimension: 360;
- KAN layer widths: `360-8-8-30`;
- Prediction horizon: 30 steps;
- Base activation function: SiLU;
- B-spline order: 3 (cubic);
- Grid size: 5;
- Grid range: `[-1, 1]`;
- Trainable parameters: 31,841.

### Physics-informed module

The training framework incorporates three physical constraints:

- Coulomb-counting consistency between SOC variation and current;
- an SOC boundary constraint for the physically meaningful SOC range;
- a monotonicity constraint between current direction and SOC variation.

### Training and transfer modules

The training objective combines the SOC prediction error with the three
physics constraints. The training framework includes gradient-based
optimization, validation-based model selection, learning-rate adjustment,
gradient clipping, and early stopping. Random seed 42 was used in the reported
experiments.

For cross-vehicle evaluation, the model trained on GZ-1 was used to initialize
the target-vehicle models. GZ-2 used partial fine-tuning, while TJ-1 used
full-model fine-tuning.

## Data volume and sample counts

The original data volume denotes the number of BMS records reported in the
manuscript. After preprocessing, one model sample consists of a 30-step input
window paired with a 30-step SOC target sequence.

| Vehicle | Original BMS records | Model samples after preprocessing |
|---|---:|---:|
| GZ-1 | 954,754 | 899,725 |
| GZ-2 | 998,243 | 922,376 |
| TJ-1 | 882,435 | 449,178 |
| **Total** | **2,835,432** | **2,271,279** |

## Data partitioning strategy

To prevent data leakage caused by overlapping sliding windows, the training,
validation, and test sets were first partitioned at the level of complete
operating segments. Sliding windows were then constructed independently within
each segment. Each sample contained 30 historical time steps and the SOC
targets for the following 30 consecutive time steps, with a stride of one.
Samples were not constructed across files or operating-segment boundaries.

In the cross-vehicle transfer experiments, GZ-1 was used as the source-domain
vehicle, while GZ-2 and TJ-1 were used as the target-domain vehicles. For GZ-2,
20% of the target-domain data were used to fine-tune the last two KAN layers.
For TJ-1, 30% of the target-domain data were used for full-model fine-tuning.


## Dependencies

The core dependencies are Python, PyTorch, NumPy, and scikit-learn.

```bash
pip install torch numpy scikit-learn
```

## References

- Z. Liu et al., "KAN: Kolmogorov-Arnold Networks," *International Conference
  on Learning Representations (ICLR)*, 2025:
  <https://proceedings.iclr.cc/paper_files/paper/2025/hash/afaed89642ea100935e39d39a4da602c-Abstract-Conference.html>
- The construction of the B-spline KAN module was implemented with reference
  to the MIT-licensed code at <https://github.com/Blealtan/efficient-kan>.
  The physics-informed constraints, multi-step SOC prediction framework,
  training procedure, and transfer-learning design were developed for this
  study.
