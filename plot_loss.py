import matplotlib.pyplot as plt
import json

# Load losses from the JSON file
with open("out/losses.json", "r") as f:
    losses = json.load(f)

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve')

# Save the plot as a PNG file
plt.savefig('out/real_losscurve.png')

# Optionally, show the plot (uncomment if you want to display the plot)
# plt.show()

print("✅ Loss curve saved as out/real_losscurve.png")

