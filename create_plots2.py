
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



file = "per_episode_metrics.csv"


model_type = ["smolvla", "xvla"]
graph_type = ["velocity", "height"]
multirun_paths = ["multirun/2025-12-04/15-14-54/", "multirun/2025-12-04/22-53-33/"]

height_words = ['up', 'down']
velocity_words = ['slow', 'fast']

def run_plots(model, graph_type):
    multirun_path = multirun_paths[model_type.index(model)]

    if graph_type == "velocity":
        words = velocity_words
        metrics_to_plot = ["min_velocity", "max_velocity", "avg_velocity"]
        names = ["Min Velocity", "Max Velocity", "Avg Velocity"]
        unit = "m/s"
    elif graph_type == "height":
        words = height_words
        metrics_to_plot = ["min_eef_height", "max_eef_height", "avg_eef_height"]
        names = ["Min End Effector Height", "Max End Effector Height", "Avg End Effector Height"]
        unit = "m"

    for name, metric in zip(names, metrics_to_plot):
        all_data = []

        multirun_word_values = {w:[] for w in words}

        for subdir in os.listdir(multirun_path):
            sub_path = os.path.join(multirun_path, subdir)
            logs_path = os.path.join(sub_path, "logs")

            if not os.path.isdir(logs_path):
                continue

            for folder in os.listdir(logs_path):
                for word in words:
                    if folder.startswith(f"{model}-{word}"):
                        csv_path = os.path.join(logs_path, folder, file)
                        df = pd.read_csv(csv_path)
                        df_success = df[df["success"] == True]
                        vals = df_success[metric].dropna().values
                        multirun_word_values[word].extend(vals)

        for word in words:
            all_data.append(multirun_word_values[word])

        for word in words:
            df = pd.read_csv(f"logs/{model}-{word}-prepend/{file}")
            df_success = df[df["success"] == True]
            all_data.append(df_success[metric].dropna().values)
        
        no_inv = pd.read_csv(f"logs/{model}-no-intervention/{file}")
        no_inv_success = no_inv[no_inv["success"] == True]
        all_data.append(no_inv_success[metric].dropna().values)

        random_inv = pd.read_csv(f"logs/{model}-random-6-a10/{file}")
        random_inv_success = random_inv[random_inv["success"] == True]
        all_data.append(random_inv_success[metric].dropna().values)

        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_data)))
        bp = plt.boxplot(all_data, tick_labels=
                    [word + " Patching" for word in words]
                     + [word + " Prepend" for word in words]
                     + ['No Intervention', 'Random Patching'], vert=True, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        plt.title(f"{model.upper()} - {metric} ({unit}) Across Conditions")
        plt.ylabel(name + f" ({unit})")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.savefig(f"plots/{model}_{metric}_boxplot.png")
        plt.close()

if sys.argv[1] == model_type[0]:
    if sys.argv[2] == graph_type[0]:
        run_plots(model_type[0], graph_type[0])
    elif sys.argv[2] == graph_type[1]:
        run_plots(model_type[0], graph_type[1])

if sys.argv[1] == model_type[1]:
    if sys.argv[2] == graph_type[0]:
        run_plots(model_type[1], graph_type[0])
    elif sys.argv[2] == graph_type[1]: 
        run_plots(model_type[1], graph_type[1])
