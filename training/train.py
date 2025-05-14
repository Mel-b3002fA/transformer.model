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
max_iters = 50_000
learning_rate = 1e-3
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

os.makedirs('out', exist_ok=True)
vocab = tokenizer.get_vocab()
stoi = vocab
itos = {idx: token for token, idx in stoi.items()}


with open('out/meta.pkl', 'wb') as f:
    pickle.dump({'vocab_size': tokenizer.vocab_size, 'stoi': stoi, 'itos': itos}, f)
print("✅ meta.pkl successfully saved.")


openwebtext = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
bookcorpus = load_dataset("bookcorpus", split="train", streaming=True, trust_remote_code=True)
commoncrawl = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
dataset = concatenate_datasets([openwebtext, bookcorpus, commoncrawl])

def tokenize_text(example):
    ids = tokenizer.encode(example['text'], truncation=True, max_length=block_size)
    if len(ids) < block_size:
        ids += [tokenizer.pad_token_id] * (block_size - len(ids))
    return torch.tensor(ids, dtype=torch.long)

tokenized_data = []
max_samples = 100_000
for idx, example in enumerate(dataset):
    try:
        ids = tokenize_text(example)
        tokenized_data.append(ids)
    except Exception as e:
        print(f"⚠️ Tokenization failed at sample {idx}: {e}")
        continue
    if idx + 1 >= max_samples:
        break


split_idx = int(0.9 * len(tokenized_data))
train_data = tokenized_data[:split_idx]
val_data = tokenized_data[split_idx:]
print(f"✅ Loaded {len(train_data)} training and {len(val_data)} validation samples.")


def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split), (batch_size,))
    x = torch.stack([data_split[i] for i in ix])
    y = x.clone()
    return x.to(device), y.to(device)


model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses = []
start_iter = 0
best_val_loss = float('inf')
ckpt_path = "out/ckpt.pt"


if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print("✅ Resumed from checkpoint.")
    if os.path.exists("out/losses.json"):
        with open("out/losses.json", "r") as f:
            losses = json.load(f)
        start_iter = len(losses)


accum_steps = 8  
loss = 0

for iter in range(max_iters):
    for _ in range(accum_steps):
        xb, yb = get_batch(...) 
        logits, curr_loss = model(xb, yb)
        curr_loss = curr_loss / accum_steps  # scale loss
        curr_loss.backward()
        loss += curr_loss.item()
    
    optimizer.step()
    optimizer.zero_grad()

    # Optional: Logging
    if iter % 100 == 0:
        print(f"iter {iter}: loss {loss:.4f}")
    loss = 0


    if iter % eval_interval == 0:
        print(f"🧪 step {iter}: train loss = {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            xb_val, yb_val = get_batch('val')
            _, val_loss = model(xb_val, yb_val)
        print(f"✅ step {iter}: val loss = {val_loss.item():.4f}")

        sample_ids = xb[0].tolist()
        decoded = tokenizer.decode(sample_ids, skip_special_tokens=True)
        print("🧠 Sample:", decoded.strip().replace("Ġ", ""))

        torch.save(model.state_dict(), ckpt_path)
        print(f"Overwrote checkpoint at {ckpt_path}")


        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), "out/best_model.pt")
            print("🌟 Best model saved at out/best_model.pt")


torch.save(model.state_dict(), ckpt_path)
print("✅ Final model checkpoint saved at", ckpt_path)


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
