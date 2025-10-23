# cognify_brain.py wala code
import numpy as np
import random
from collections import defaultdict

class DigitalBrain:
    """
    Simulated 'brain twin' that learns behavioral transitions
    and predicts next actions.
    """

    def __init__(self, actions):
        self.actions = actions
        self.transition_matrix = defaultdict(lambda: defaultdict(int))

    def learn(self, sequence):
        """Learn action transitions from behavior sequence."""
        for i in range(len(sequence) - 1):
            curr, nxt = sequence[i], sequence[i + 1]
            self.transition_matrix[curr][nxt] += 1

        # Normalize to probabilities (last mein)
        for curr in self.transition_matrix:
            total = sum(self.transition_matrix[curr].values())
            for nxt in self.transition_matrix[curr]:
                self.transition_matrix[curr][nxt] /= total

    def predict_next(self, current_action):
        """Predict next action based on learned probabilities."""
        if current_action not in self.transition_matrix:
            return random.choice(self.actions)

        probs = self.transition_matrix[current_action]
        next_actions = list(probs.keys())
        weights = list(probs.values())

        return random.choices(next_actions, weights=weights, k=1)[0]

    def visualize_matrix(self):
        """Return transition matrix as a 2D numpy array for visualization."""
        matrix = np.zeros((len(self.actions), len(self.actions)))
        for i, a1 in enumerate(self.actions):
            for j, a2 in enumerate(self.actions):
                matrix[i, j] = self.transition_matrix[a1][a2] if a2 in self.transition_matrix[a1] else 0
        return matrix
