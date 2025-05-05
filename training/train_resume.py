import sys
import os
import torch
import pickle
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig
from model.tokenizer import Tokenizer


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

# === Checkpoint Handling ===
checkpoint_path = 'out/ckpt.pt'
start_epoch = 0  # Default to starting at epoch 0

if os.path.exists(checkpoint_path):
    print(f"Checkpoint found at {checkpoint_path}, resuming from last saved epoch.")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch after the checkpoint
else:
    print("No checkpoint found, starting from scratch.")

# === Training Loop ===
losses = []

for iter in range(start_epoch, max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if iter % eval_interval == 0:
        print(f"step {iter}: loss = {loss.item():.4f}")

# === Save Model & Loss ===
checkpoint = {
    'epoch': iter,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, "out/ckpt.pt")
print("✅ Model checkpoint saved at out/ckpt.pt")


with open("out/losses.json", "w") as f:
    json.dump(losses, f)
print("✅ Losses saved to out/losses.json")


plt.figure(figsize=(10, 6))
plt.plot(losses, label="Training Loss", color="blue", linewidth=2)
plt.xlabel("Training Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()


plt.savefig("out/second_losscurve.png")
print("✅ Loss curve saved as out/loss_curve.png")
