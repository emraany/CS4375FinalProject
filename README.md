# CS 4375 Final Project — CNN Comparison

Phase 1 pipeline for training and evaluating CNN architectures on MNIST and
CIFAR-10. Models included are LeNet, a from-scratch ResNet-18 adapted for
32x32 inputs, and SE-ResNet-18. SE-ResNet-18 is a ResNet-18 variant with
squeeze-and-excitation channel attention inside each residual block.

Supported model names: `lenet`, `resnet18`, `se_resnet18`.

## Project layout

```
main.py                  # run(model, dataset, **kwargs)
data/loaders.py          # MNIST and CIFAR-10 loaders, 90/10 train/val split
models/lenet.py          # classic LeNet-5 (ReLU, MaxPool)
models/resnet18.py       # ResNet-18 from scratch, CIFAR stem
models/se_resnet18.py    # SE-ResNet-18 with channel attention
models/__init__.py       # MODEL_REGISTRY and build_model
training/trainer.py      # train loop with TensorBoard scalars
evaluation/evaluator.py  # test metrics, confusion matrix, curve plots
utils/seed.py            # set_seed
```

Outputs (created at runtime):

```
results/{model}_{dataset}/
    loss_curve.png  accuracy_curve.png  confusion_matrix.png
    metrics.json    weights.pt
runs/{model}_{dataset}/   # TensorBoard event files
datasets/                 # torchvision downloads land here
```

## Running on Colab

Upload the zipped project to Drive, mount, and unzip into the working dir.

Cell 1: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Cell 2: Unzip and move into project

```python
!unzip -qo /content/drive/MyDrive/mlfinal.zip -d /content/mlfinal
%cd /content/mlfinal
```

Cell 3: Install requirements

```python
!pip install -q -r requirements.txt
```

Cell 4: Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

Cell 5: Smoke test

```python
from main import run

result = run("lenet", "mnist", epochs=1)
print(result)
```

Cell 6: Full training runs

```python
from main import run

results = {}
for model in ["lenet", "resnet18", "se_resnet18"]:
    for dataset in ["mnist", "cifar10"]:
        print(f"\n{'='*50}")
        print(f"Training {model} on {dataset}")
        print(f"{'='*50}")
        results[f"{model}_{dataset}"] = run(model, dataset)
        print(f"Done: {results[f'{model}_{dataset}']}")
```

Override hyperparameters as kwargs, e.g. `run("resnet18", "cifar10", epochs=50, lr=5e-4)`.

Cell 7: TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

Four scalars per run: `loss/train`, `loss/val`, `acc/train`, `acc/val`.

Cell 8: Check all results landed

```python
import os
for run_dir in sorted(os.listdir("results")):
    files = os.listdir(f"results/{run_dir}")
    print(f"{run_dir}: {sorted(files)}")
```

Cell 9: Download results to Drive

```python
!cp -r results /content/drive/MyDrive/mlfinal_results
!cp -r runs /content/drive/MyDrive/mlfinal_runs
print("Results saved to Drive")
```

## Defaults

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | CrossEntropyLoss |
| Batch size | 128 |
| Epochs | 25 |
| Val split | 10% |
| Seed | 42 |

## Adding a new model

1. Create `models/your_model.py` with a class inheriting `nn.Module` and the
   constructor signature `(in_channels, num_classes)`.
2. Add it to `MODEL_REGISTRY` in `models/__init__.py`.
3. Call `run("your_model", "cifar10")`.

No changes to training, evaluation, or data code are needed.
