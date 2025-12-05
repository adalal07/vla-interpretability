import os
import re
import pandas as pd
import matplotlib.pyplot as plt

root_dir = "multirun/2025-12-04/15-14-54/"

pattern = re.compile(r"smolvla-(.*)-(\d+)-([\d\.]+)")

records = []

metrics_to_plot = [
        "max_velocity", "avg_velocity",
        "min_eef_height", "max_eef_height", "avg_eef_height"
    ]

for subdir in os.listdir(root_dir):
    sub_path = os.path.join(root_dir, subdir)
    if not os.path.isdir(sub_path):
        continue

    logs_path = os.path.join(sub_path, "logs")

    for folder in os.listdir(logs_path):
        match = pattern.match(folder)
        if not match:
            continue

        target = match.groups()[0]
        k = int(match.groups()[1])
        alpha = float(match.groups()[2])

        csv_path = os.path.join(logs_path, folder, "per_episode_metrics.csv")

        df = pd.read_csv(csv_path)
        df_success = df[df["success"] == True]

        metrics = {
            "target": target,
            "k": k,
            "alpha": alpha,
        }

        for col in metrics_to_plot:
            metrics[col] = df_success[col].mean()

        records.append(metrics)
    
    data = pd.DataFrame(records)

    os.makedirs("plots", exist_ok=True)

    for target in data["target"].unique():
        subdf = data[data["target"] == target]

        for metric in metrics_to_plot:
            plt.figure()
            pivot = subdf.pivot_table(
                index="k", columns="alpha", values=metric
            )

            plt.xlabel("k")
            plt.ylabel(metric)
            plt.title(f"{metric} for target={target}")

            plt.plot(pivot.index, pivot.values)
            plt.legend([f"alpha={col}" for col in pivot.columns])

            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"plots/smolvla/{metric}_target={target}.png")
            plt.close()

    word_stats = (
        data.groupby("target")[metrics_to_plot]
        .agg(['mean', 'std'])
        .reset_index()
    )

    for metric in metrics_to_plot:
        plt.figure()
        plt.xlabel("target")
        plt.ylabel(metric)
        plt.title(f"Comparison of {metric} across targets")

        means = word_stats[(metric, 'mean')]
        stds = word_stats[(metric, 'std')]
        words = word_stats["target"]

        plt.bar(
            words,
            means,
            yerr=stds,
            capsize=5,
        )

        y_min = (means - stds).min() * 0.95 if not ((means - stds).min() is pd.NA) else 0
        y_max = (means + stds).max() * 1.05 if not ((means + stds).max() is pd.NA) else 1
        if pd.isna(y_min) or pd.isna(y_max) or y_min >= y_max:
            y_min, y_max = 0, 1   
        plt.ylim(y_min, y_max)

        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/smolvla/compare_{metric}.png")
        plt.close()