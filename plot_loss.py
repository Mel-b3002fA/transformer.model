import json
import matplotlib.pyplot as plt

# Load losses
with open("out/losses.json", "r") as f:
    losses = json.load(f)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(losses, label='Training Loss')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
