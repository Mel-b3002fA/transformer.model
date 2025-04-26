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
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import T5Tokenizer
from model.transformer import GPT, GPTConfig

# Step 1: Load the WMT 2014 dataset (DE-EN)
dataset = load_dataset("wmt14", "de-en")

# Limit to first 100 examples for quick testing
train_data = dataset["train"].select(range(200))
valid_data = dataset["validation"]
test_data = dataset["test"]

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token if missing

# Tokenization function
def tokenize(text):
    return tokenizer.encode(text, truncation=True, padding="max_length", max_length=128)

# Model configuration
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    n_embd=64,
    n_head=4,
    n_layer=4
)
model = GPT(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
losses = []

# Training loop
for step, example in enumerate(train_data):
    if step >= 700:
        break

    input_text = example["translation"]["en"]
    target_text = example["translation"]["de"]

    input_tensor = torch.tensor([tokenize(input_text)]).to(device)
    target_tensor = torch.tensor([tokenize(target_text)]).to(device)

    model.train()
    optimizer.zero_grad()

    logits, loss = model(input_tensor, target_tensor)

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Plot the loss curve
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.savefig("loss_curve.png")

# Evaluation with a sample from validation set
model.eval()
with torch.no_grad():
    example = valid_data[0]
    input_text = example["translation"]["en"]

    input_tensor = torch.tensor([tokenize(input_text)]).to(device)
    logits, _ = model(input_tensor)
    prediction = torch.argmax(logits, dim=-1)

    print("\nSample Prediction:")
    print("Input:     ", tokenizer.decode(input_tensor[0], skip_special_tokens=True))
    print("Predicted: ", tokenizer.decode(prediction[0], skip_special_tokens=True))
# Evaluation with autoregressive decoding
model.eval()
with torch.no_grad():
    example = valid_data[0]
    input_text = example["translation"]["en"]
    input_ids = torch.tensor([tokenize(input_text)]).to(device)

    max_new_tokens = 50
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        logits, _ = model(generated)
        next_token_logits = logits[:, -1, :]  # Get logits for last generated token
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # Greedy decoding

        generated = torch.cat((generated, next_token), dim=1)

        # Stop if model generates a pad or end token
        if next_token.item() in [tokenizer.pad_token_id, tokenizer.eos_token_id]:
            break

    print("\nSample Prediction (Greedy):")
    print("Input:     ", input_text)
    print("Predicted: ", tokenizer.decode(generated[0], skip_special_tokens=True))
