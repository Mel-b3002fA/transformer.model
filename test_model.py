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






















""" import torch
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


""" """


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.transformer import GPT, GPTConfig


config = GPTConfig(
    vocab_size=1000,
    n_embd=64,
    n_head=4,
    n_layer=4
)

model = GPT(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


batch_size = 8
seq_len = 16
x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
y = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


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


plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.savefig("loss_curve.png")  


model.eval()
with torch.no_grad():
    test_input = x[0].unsqueeze(0) 
    logits, _ = model(test_input)
    prediction = torch.argmax(logits, dim=-1)

    print("\nSample Prediction:")
    print("Input:     ", test_input[0].tolist())
    print("Predicted: ", prediction[0].tolist())
 """




import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from model.transformer import GPT, GPTConfig
from data.toy_data.toy_data import toy_data

USE_TOY_DATA = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_config(use_toy=True):
    if use_toy:
        return GPTConfig(
            vocab_size=10,
            block_size=4,
            n_layer=2,
            n_head=2,
            n_embd=64
        )
    else:
        return GPTConfig(
            vocab_size=1000,
            block_size=16,
            n_layer=4,
            n_head=4,
            n_embd=64
        )


def train_model(model, optimizer, config, use_toy=True, steps=100):
    model.train()
    all_losses = []

    for step in range(steps):
        total_loss = 0.0

        if use_toy:
            for seq in toy_data:
                x = torch.tensor(seq[:-1]).unsqueeze(0).to(DEVICE)
                y = torch.tensor(seq[1:]).unsqueeze(0).to(DEVICE)

                logits, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        else:
            batch_size = 8
            seq_len = config.block_size
            x = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(DEVICE)
            y = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(DEVICE)

            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = loss.item()

        all_losses.append(total_loss)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {total_loss:.4f}")
    return all_losses


def evaluate_model(model, config, use_toy=True):
    model.eval()
    with torch.no_grad():
        if use_toy:
            test_input = torch.tensor(toy_data[0][:-1]).unsqueeze(0).to(DEVICE)
        else:
            test_input = torch.randint(0, config.vocab_size, (1, config.block_size)).to(DEVICE)

        logits, _ = model(test_input)
        prediction = torch.argmax(logits, dim=-1)

        print("\nSample Prediction:")
        print("Input:     ", test_input[0].tolist())
        print("Predicted: ", prediction[0].tolist())


def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.savefig("loss_curve.png")


def main():
    config = create_config(USE_TOY_DATA)
    model = GPT(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    losses = train_model(model, optimizer, config, use_toy=USE_TOY_DATA)
    evaluate_model(model, config, use_toy=USE_TOY_DATA)
    plot_losses(losses)

if __name__ == "__main__":
    main()
