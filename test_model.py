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


""" 


""" 
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
    print("Predicted: ", prediction[0].tolist()) """




# not working bc of toy data syntax
""" import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model.transformer import GPT, GPTConfig
from data.toy_data.toy_data import toy_data  


USE_TOY_DATA = True  # Toggle between toy data and random data
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
            for seq in toy_data():

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
            x = torch

# imports, config functions, training, evaluation, plotting...

def main():
    config = create_config(USE_TOY_DATA)
    model = GPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = train_model(model, optimizer, config, use_toy=USE_TOY_DATA)
    evaluate_model(model, config, use_toy=USE_TOY_DATA)
    plot_losses(losses)


if __name__ == "__main__":
    main() 


 """




import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import T5Tokenizer
from model.transformer import GPT, GPTConfig
from torch.utils.data import DataLoader

# Step 1: Load the WMT 2014 dataset
dataset = load_dataset("wmt14", "de-en")

# Limit to first 200 examples for faster testing
train_data = dataset["train"].select(range(200))
valid_data = dataset["validation"]

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_text_pair(example):
    input_ids = tokenizer.encode(example["translation"]["en"], truncation=True, padding="max_length", max_length=128)
    target_ids = tokenizer.encode(example["translation"]["de"], truncation=True, padding="max_length", max_length=128)
    return {"input_ids": input_ids, "target_ids": target_ids}

train_data = train_data.map(tokenize_text_pair)

# Create DataLoader
batch_size = 16

def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    target_ids = torch.tensor([item["target_ids"] for item in batch])
    return input_ids, target_ids

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model configuration
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    n_embd=128,   # slight increase in model size
    n_head=4,
    n_layer=4
)
model = GPT(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

losses = []

# Training loop
num_steps = 500

step = 0
while step < num_steps:
    for batch in train_loader:
        input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        model.train()
        optimizer.zero_grad()

        logits, loss = model(input_ids, target_ids)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        step += 1
        if step >= num_steps:
            break

# Plot the loss curve
plt.figure(figsize=(10,6))
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.savefig("loss_curve.png")

# Sample prediction (autoregressive decoding)
model.eval()
with torch.no_grad():
    example = valid_data[0]
    input_text = example["translation"]["en"]
    input_ids = torch.tensor([tokenizer.encode(input_text, truncation=True, padding="max_length", max_length=128)]).to(device)

    generated = input_ids.clone()

    max_new_tokens = 50

    for _ in range(max_new_tokens):
        logits, _ = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        generated = torch.cat((generated, next_token), dim=1)

        if next_token.item() in [tokenizer.pad_token_id, tokenizer.eos_token_id]:
            break

    print("\nSample Prediction (Greedy decoding):")
    print("Input:     ", input_text)
    print("Predicted: ", tokenizer.decode(generated[0], skip_special_tokens=True))
