""" from model.transformer import GPT, GPTConfig

# Define a small test config
config = GPTConfig(
    n_layer=2,
    n_head=2,
    n_embd=128,
    block_size=64
)

model = GPT(config)
import torch

# Batch of 4 sequences, each 64 tokens long
x = torch.randint(0, config.vocab_size, (4, config.block_size))

# Forward pass
out = model(x)

print("Output shape:", out.shape) """



import torch
from model.transformer import GPT, GPTConfig
from data.toy_data.toy_data import toy_data 

config = GPTConfig(
    vocab_size=10,
    block_size=4,
    n_layer=2,
    n_head=2,
    n_embd=64
)

model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


model.train()
for step in range(100):
    total_loss = 0.0
    for seq in toy_data:
        x = torch.tensor(seq[:-1]).unsqueeze(0)  
        y = torch.tensor(seq[1:]).unsqueeze(0)  

        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {total_loss:.4f}")









import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.transformer import GPT, GPTConfig

# Configuration
config = GPTConfig(
    vocab_size=1000,
    n_embd=64,
    n_head=4,
    n_layer=4
)

model = GPT(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Dummy training data
batch_size = 8
seq_len = 16
x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
y = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
losses = []

for step in range(100):
    model.train()
    optimizer.zero_grad()

    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Plot loss
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.savefig("loss_curve.png")  # or plt.show() if running locally

# Evaluation / Generation (optional)
model.eval()
with torch.no_grad():
    test_input = x[0].unsqueeze(0)  # one sample
    logits, _ = model(test_input)
    prediction = torch.argmax(logits, dim=-1)

    print("\nSample Prediction:")
    print("Input:     ", test_input[0].tolist())
    print("Predicted: ", prediction[0].tolist())
