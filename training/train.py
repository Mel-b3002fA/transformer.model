import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pickle
import json
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from model import GPT, GPTConfig

batch_size = 4
block_size = 128
max_iters = 1000
learning_rate = 1e-3
eval_interval = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load datasets
openwebtext = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
bookcorpus = load_dataset("bookcorpus", split="train", streaming=True, trust_remote_code=True)
commoncrawl = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)



# Concatenate datasets
dataset = concatenate_datasets([openwebtext, bookcorpus, commoncrawl])

def tokenize(example):
    return tokenizer(example["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=block_size)

tokenized_stream = (tokenize(sample) for sample in dataset)

os.makedirs('out', exist_ok=True)
vocab = tokenizer.get_vocab()
stoi = vocab
itos = {idx: token for token, idx in stoi.items()}
with open('out/meta.pkl', 'wb') as f:
    pickle.dump({'vocab_size': tokenizer.vocab_size, 'stoi': stoi, 'itos': itos}, f)
print("✅ meta.pkl successfully saved.")

train_data = []
val_data = []
for idx, tokenized_example in enumerate(tokenized_stream):
    if idx < 0.9 * max_iters:
        train_data.append(tokenized_example)
    else:
        val_data.append(tokenized_example)
    if idx >= max_iters:
        break

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split), (batch_size,))
    x = torch.stack([data_split[i]["input_ids"].squeeze(0) for i in ix])  
    y = torch.stack([data_split[i]["input_ids"].squeeze(0) for i in ix])
    return x.to(device), y.to(device)

model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses = []
start_iter = 0
ckpt_path = "out/ckpt.pt"
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print("✅ Resumed from checkpoint.")
    if os.path.exists("out/losses.json"):
        with open("out/losses.json", "r") as f:
            losses = json.load(f)
        start_iter = len(losses)

for iter in range(start_iter, max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if iter % eval_interval == 0:
        print(f"step {iter}: loss = {loss.item():.4f}")

torch.save(model.state_dict(), "out/ckpt.pt")
print("✅ Model checkpoint saved at out/ckpt.pt")

with open("out/losses.json", "w") as f:
    json.dump(losses, f)
print("✅ Losses saved to out/losses.json")

plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig("out/bigdata_loss.png")
plt.show()
