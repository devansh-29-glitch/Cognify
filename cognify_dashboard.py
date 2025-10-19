# cognify_dashboard.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_transition_matrix(matrix, actions, save_path):
    """
    Visualize the brain's learned transition probabilities as a heatmap.
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(matrix, annot=True, cmap="viridis", xticklabels=actions, yticklabels=actions)
    plt.title("🧠 Cognify — Digital Brain Transition Map")
    plt.xlabel("Next Action →")
    plt.ylabel("Current Action →")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"🧩 Saved neural connection heatmap to: {save_path}")
