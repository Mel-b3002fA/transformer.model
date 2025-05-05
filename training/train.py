import sys
import os
import torch
import pickle
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from model import GPT, GPTConfig
from model.tokenizer import Tokenizer

# Configuration
batch_size = 4
block_size = 128
max_iters = 200
learning_rate = 1e-3
eval_interval = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Ensure padding token is set (use eos_token as pad_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the OpenWebText dataset
dataset = load_dataset("openwebtext", split="train", streaming=True)

# Tokenization function
def tokenize(example):
    return tokenizer(example["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=block_size)

# Tokenize on the fly
tokenized_stream = (tokenize(sample) for sample in dataset)

# Save tokenizer and metadata
os.makedirs('out', exist_ok=True)

# Use the tokenizer's get_vocab() method to access the token-to-index (stoi) and index-to-token (itos) mappings
vocab = tokenizer.get_vocab()
stoi = vocab
itos = {idx: token for token, idx in stoi.items()}

with open('out/meta.pkl', 'wb') as f:
    pickle.dump({
        'vocab_size': tokenizer.vocab_size,
        'stoi': stoi,
        'itos': itos
    }, f)
print("✅ meta.pkl successfully saved.")

# Prepare training and validation splits
# Since we are streaming data, we can't pre-define the split as done in traditional datasets.
# Instead, we manually split the data.
train_data = []
val_data = []

# Simulate a 90/10 split by iterating over the first 90% for training and the remaining 10% for validation
for idx, tokenized_example in enumerate(tokenized_stream):
    if idx < 0.9 * max_iters:  # Train on the first 90%
        train_data.append(tokenized_example)
    else:  # Validate on the remaining 10%
        val_data.append(tokenized_example)
    
    # Break the loop once we've reached the desired number of iterations
    if idx >= max_iters:
        break

# Prepare batch data for training
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i]["input_ids"].squeeze(1) for i in ix])
    y = torch.stack([data_split[i]["input_ids"].squeeze(1) for i in ix])  # Predict the next token
    return x.to(device), y.to(device)

# Initialize the model
model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Losses for tracking
losses = []
start_iter = 0

# Load checkpoint if it exists
ckpt_path = "out/ckpt.pt"
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print("✅ Resumed from checkpoint.")
    if os.path.exists("out/losses.json"):
        with open("out/losses.json", "r") as f:
            losses = json.load(f)
        start_iter = len(losses)

# Training loop
for iter in range(start_iter, max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if iter % eval_interval == 0:
        print(f"step {iter}: loss = {loss.item():.4f}")

# Save model and losses
os.makedirs("out", exist_ok=True)
torch.save(model.state_dict(), "out/ckpt.pt")
print("✅ Model checkpoint saved at out/ckpt.pt")

with open("out/losses.json", "w") as f:
    json.dump(losses, f)
print("✅ Losses saved to out/losses.json")
