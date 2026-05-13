import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
OUT_PNG = Path("results_table.png")
OUT_LATEX = Path("finaltablelatex.tex")


def pretty_model(name):
    names = {
        "lenet": "LeNet",
        "resnet18": "ResNet-18",
        "se_resnet18": "SE-ResNet-18",
    }
    return names.get(name, name)


def pretty_dataset(name):
    names = {
        "mnist": "MNIST",
        "cifar10": "CIFAR-10",
    }
    return names.get(name, name)


def parse_run_name(run_name):
    if run_name.endswith("_cifar10"):
        return run_name.replace("_cifar10", ""), "cifar10"
    if run_name.endswith("_mnist"):
        return run_name.replace("_mnist", ""), "mnist"
    return run_name, "unknown"


def fmt(x):
    if x is None:
        return ""
    return f"{x:.4f}"


def make_latex_table(columns, rows):
    latex = []

    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Final Model Comparison on MNIST and CIFAR-10}")
    latex.append("\\label{tab:final_model_comparison}")
    latex.append("\\begin{tabular}{llcccccc}")
    latex.append("\\hline")
    latex.append(" & ".join(columns) + " \\\\")
    latex.append("\\hline")

    for row in rows:
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


rows = []

for run_folder in sorted(RESULTS_DIR.iterdir()):
    if not run_folder.is_dir():
        continue

    metrics_path = run_folder / "metrics.json"
    if not metrics_path.exists():
        continue

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    model_name, dataset_name = parse_run_name(run_folder.name)
    final_epoch = metrics.get("final_epoch", {})

    rows.append([
        pretty_model(model_name),
        pretty_dataset(dataset_name),
        fmt(metrics["test_accuracy"]),
        fmt(metrics["macro"]["precision"]),
        fmt(metrics["macro"]["recall"]),
        fmt(metrics["macro"]["f1"]),
        fmt(metrics["weighted"]["f1"]),
        fmt(final_epoch.get("best_val_acc")),
    ])


model_order = {"LeNet": 0, "ResNet-18": 1, "SE-ResNet-18": 2}
dataset_order = {"MNIST": 0, "CIFAR-10": 1}

rows.sort(key=lambda r: (model_order.get(r[0], 99), dataset_order.get(r[1], 99)))


columns = [
    "Model",
    "Dataset",
    "Test Acc.",
    "Macro Prec.",
    "Macro Rec.",
    "Macro F1",
    "Weighted F1",
    "Best Val Acc.",
]


# Save LaTeX table
latex_table = make_latex_table(columns, rows)

with open(OUT_LATEX, "w") as f:
    f.write(latex_table)

print(f"Saved {OUT_LATEX}")


# Save PNG table
fig, ax = plt.subplots(figsize=(13, 3.2))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Bold header row
for col in range(len(columns)):
    table[(0, col)].set_text_props(weight="bold")

plt.title("Final Model Comparison on MNIST and CIFAR-10", fontsize=14, weight="bold", pad=16)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved {OUT_PNG}")