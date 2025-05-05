import sys
import os
import torch
import pickle
import json
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig
from model.tokenizer import Tokenizer

# Hyperparameters
batch_size = 4
block_size = 128
max_iters = 200
learning_rate = 1e-3
eval_interval = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer
tokenizer = Tokenizer()

# Stream The Pile dataset and tokenize on the fly
streaming_dataset = load_dataset("the_pile", split="train", streaming=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize(example):
    return tokenizer(example["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=block_size)

# Create tokenized generator for streaming data
tokenized_stream = (tokenize(sample) for sample in streaming_dataset)

# Save tokenizer metadata
os.makedirs('out', exist_ok=True)
with open('out/meta.pkl', 'wb') as f:
    pickle.dump({
        'vocab_size': tokenizer.vocab_size,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos
    }, f)
print("✅ meta.pkl successfully saved.")

# Initialize model
model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Check for existing checkpoint
ckpt_path = "out/ckpt.pt"
losses = []
start_iter = 0
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print("✅ Resumed from checkpoint.")
    if os.path.exists("out/losses.json"):
        with open("out/losses.json", "r") as f:
            losses = json.load(f)
        start_iter = len(losses)

# Function to generate batches from the tokenized stream
def get_batch(data_stream):
    batch = list(islice(data_stream, batch_size))
    input_ids = torch.stack([item["input_ids"].squeeze(1) for item in batch])
    attention_mask = torch.stack([item["attention_mask"].squeeze(1) for item in batch])
    return input_ids.to(device), attention_mask.to(device)

# Training loop
from itertools import islice

for iter in range(start_iter, max_iters):
    xb, yb = get_batch(tokenized_stream)  # Get batch from the streaming dataset
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if iter % eval_interval == 0:
        print(f"step {iter}: loss = {loss.item():.4f}")

    # Save checkpoint periodically
    if iter % 100 == 0:
        os.makedirs("out", exist_ok=True)
        torch.save(model.state_dict(), "out/ckpt.pt")
        print("✅ Model checkpoint saved at out/ckpt.pt")

# Save losses to track progress
with open("out/losses.json", "w") as f:
    json.dump(losses, f)
print("✅ Losses saved to out/losses.json")
