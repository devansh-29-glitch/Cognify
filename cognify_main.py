# cognify_main.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Create data folder if missing
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ACTIONS = ["wake", "walk", "sit", "type", "think", "eat", "sleep"]

def generate_behavior_sequence(n_steps=1000, seed=42):
    """
    Generate a synthetic sequence of human-like actions.
    The probability of next action depends weakly on the previous one.
    """
    random.seed(seed)
    seq = ["wake"]

    transition_probs = {
        "wake": ["walk", "think", "eat"],
        "walk": ["sit", "think", "eat"],
        "sit": ["type", "think", "sleep"],
        "type": ["think", "walk", "sleep"],
        "think": ["type", "eat", "sleep"],
        "eat": ["walk", "sit", "sleep"],
        "sleep": ["wake"]
    }

    for _ in tqdm(range(n_steps), desc="🧠 Simulating behavior sequence"):
        last = seq[-1]
        next_action = random.choice(transition_probs[last])
        seq.append(next_action)

    return seq

def save_sequence(seq, path):
    with open(path, "w") as f:
        for step in seq:
            f.write(step + "\n")

def plot_action_distribution(seq, save_path):
    plt.figure(figsize=(8,4))
    sns.countplot(x=seq, order=ACTIONS)
    plt.title("Action Frequency — Simulated Human Behavior")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    sequence = generate_behavior_sequence(n_steps=500)
    out_path = DATA_DIR / "behavior_sequence.csv"
    save_sequence(sequence, out_path)
    print(f"\n✅ Saved simulated behavior data to: {out_path}")

    # Visualize
    ASSETS = Path("assets")
    ASSETS.mkdir(exist_ok=True)
    plot_path = ASSETS / "action_distribution.png"
    plot_action_distribution(sequence, plot_path)
    print(f"📊 Saved visualization to: {plot_path}")

    # --- Test the Digital Brain ---
    from cognify_brain import DigitalBrain

    brain = DigitalBrain(ACTIONS)
    brain.learn(sequence)

    current_action = random.choice(ACTIONS)
    predicted_action = brain.predict_next(current_action)

    print(f"\n🧠 Digital Brain says: after '{current_action}', you're most likely to '{predicted_action}' next.")

    # --- Visualize the brain’s learned transition matrix ---
    from cognify_dashboard import plot_transition_matrix

    matrix = brain.visualize_matrix()
    plot_transition_matrix(matrix, ACTIONS, "assets/transition_matrix.png")