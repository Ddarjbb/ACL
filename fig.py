import os
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def load_json_files(directory):
    json_files = glob(os.path.join(directory, "*.json"))
    data_list = []
    for file in json_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data_list.append((os.path.basename(file), data))
    return data_list

def plot_radar(ax, data):
    labels = ["O", "A", "C", "N", "E"]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    for agent, values in data.items():
        values_list = [values[label] for label in labels]
        values_list += values_list[:1]  # Close the circle
        ax.plot(angles, values_list, label=agent)
        ax.fill(angles, values_list, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

def main():
    directory = "/mnt/data/guoxin/Big5LLMTestFigs"
    data_list = load_json_files(directory)
    
    num_files = len(data_list)
    cols = 2  # Set to 2 columns
    rows = (num_files // cols) + (1 if num_files % cols else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), subplot_kw={'polar': True})
    axes = axes.flatten()
    
    legend_labels = set()
    for i, (filename, data) in enumerate(data_list):
        plot_radar(axes[i], data)
        legend_labels.update(data.keys())
        axes[i].set_title(os.path.splitext(filename)[0])
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("radar_chart.png")
if __name__ == "__main__":
    main()
