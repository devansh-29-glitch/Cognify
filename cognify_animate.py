# cognify_animate.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from pathlib import Path
from cognify_brain import DigitalBrain
from cognify_main import ACTIONS, generate_behavior_sequence

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

# Generate a sequence
sequence = generate_behavior_sequence(n_steps=100)

# Initialize brain
brain = DigitalBrain(ACTIONS)

# Prepare figure
fig, ax = plt.subplots(figsize=(8,6))

def update(frame):
    """Update the heatmap at each learning step."""
    brain.learn(sequence[:frame + 2])
    matrix = brain.visualize_matrix()
    ax.clear()
    sns.heatmap(matrix, annot=False, cmap="viridis", xticklabels=ACTIONS, yticklabels=ACTIONS, ax=ax)
    ax.set_title(f"🧠 Cognify Learning — Step {frame+1}/{len(sequence)}")
    ax.set_xlabel("Next Action →")
    ax.set_ylabel("Current Action →")

ani = FuncAnimation(fig, update, frames=len(sequence)//5, interval=200)
gif_path = ASSETS / "cognify_learning.gif"
ani.save(gif_path, writer="pillow", fps=5)

print(f"🎬 Animated brain learning process saved at: {gif_path}")