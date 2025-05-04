import matplotlib.pyplot as plt
import json

with open("out/losses.json", "r") as f:
    losses = json.load(f)


plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve')

plt.savefig('out/real_losscurve.png')


print("✅ Loss curve saved as out/real_losscurve.png")

