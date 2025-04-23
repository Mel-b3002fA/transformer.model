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


# test_model.py
import torch
from model.transformer import GPT, GPTConfig
from toy_data import toy_data

# Config for small test
config = GPTConfig(
    vocab_size=10,
    block_size=4,
    n_layer=2,
    n_head=2,
    n_embd=64
)

model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
model.train()
for step in range(100):
    total_loss = 0.0
    for seq in toy_data:
        x = torch.tensor(seq[:-1]).unsqueeze(0)  # input
        y = torch.tensor(seq[1:]).unsqueeze(0)   # target

        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {total_loss:.4f}")

