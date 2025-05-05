import sys
import os
import torch
import pickle
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig
from model.tokenizer import Tokenizer

# === Config ===
batch_size = 4
block_size = 128
max_iters = 200
learning_rate = 1e-3
eval_interval = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load data ===
with open('data/joi.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# === Tokenizer ===
tokenizer = Tokenizer()
tokenizer.train(text)
vocab_size = tokenizer.vocab_size

# Save metadata
os.makedirs('out', exist_ok=True)
with open('out/meta.pkl', 'wb') as f:
    pickle.dump({
        'vocab_size': tokenizer.vocab_size,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos
    }, f)
print("✅ meta.pkl successfully saved.")

# === Encode dataset ===
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
train_data = data[:int(0.9 * len(data))]
val_data = data[int(0.9 * len(data)):]

# === Batch function ===
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# === Model ===
model = GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# === Training Loop ===
losses = []

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if iter % eval_interval == 0:
        print(f"step {iter}: loss = {loss.item():.4f}")

# === Save Model & Loss ===
torch.save(model.state_dict(), "out/ckpt.pt")
print("✅ Model checkpoint saved at out/ckpt.pt")

# Save losses to file for potential future use
with open("out/losses.json", "w") as f:
    json.dump(losses, f)
print("✅ Losses saved to out/losses.json")

# === Plotting the Loss Curve ===
plt.figure(figsize=(10, 6))
plt.plot(losses, label="Training Loss", color="blue", linewidth=2)
plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("out/second_losscurve.png")
print("✅ Loss curve saved as out/loss_curve.png")

# Optionally, show the plot (this will pop up a window with the graph)
# plt.show()
