""" import sys
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
 """








import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pickle
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from model.gpt import GPT, GPTConfig
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameters
batch_size = 4
block_size = 128
max_iters = 10_000
learning_rate = 1e-4
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gradient_clip = 1.0  # Add gradient clipping
accum_steps = 8

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create output directory
os.makedirs('out', exist_ok=True)

# Save vocabulary metadata
vocab = tokenizer.get_vocab()
stoi = vocab
itos = {idx: token for token, idx in stoi.items()}
with open('out/meta.pkl', 'wb') as f:
    pickle.dump({'vocab_size': tokenizer.vocab_size, 'stoi': stoi, 'itos': itos}, f)
logger.info("✅ meta.pkl successfully saved.")

# Load and preprocess dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_instruction(example):
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {'text': text}

formatted_dataset = dataset.map(format_instruction)

def tokenize_text(example):
    ids = tokenizer.encode(example['text'], truncation=True, max_length=block_size)
    if len(ids) < block_size:
        ids += [tokenizer.pad_token_id] * (block_size - len(ids))
    return {'input_ids': ids}

tokenized_dataset = formatted_dataset.map(tokenize_text)
tokenized_data = [torch.tensor(x, dtype=torch.long) for x in tokenized_dataset['input_ids']]

# Split dataset
split_idx = int(0.9 * len(tokenized_data))
train_data = tokenized_data[:split_idx]
val_data = tokenized_data[split_idx:]
logger.info(f"✅ Loaded {len(train_data)} training and {len(val_data)} validation samples.")

# Log sample data for debugging
sample_data = tokenized_data[0][:20]  # First 20 tokens of first sample
logger.info(f"Sample tokenized data: {sample_data.tolist()}")
logger.info(f"Sample decoded: {tokenizer.decode(sample_data, skip_special_tokens=True)}")

# Batch function
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split), (batch_size,))
    x = torch.stack([data_split[i] for i in ix])
    y = x.clone()
    return x.to(device), y.to(device)

# Initialize model
config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size, n_layer=4, n_head=4, n_embd=128)
model = GPT(config).to(device)

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

initialize_weights(model)
logger.info("✅ Model weights initialized with Xavier and normal distributions.")
logger.warning("⚠️ Model lacks Transformer blocks, which may lead to poor performance and nonsense outputs.")

# Log model configuration
logger.info(f"Model config: vocab_size={config.vocab_size}, block_size={config.block_size}, "
            f"n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")
logger.warning("⚠️ Note: n_layer and n_head are unused in the current GPT model due to missing Transformer blocks.")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses = []
start_iter = 0
best_val_loss = float('inf')
ckpt_path = f"out/ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
best_model_path = "out/best_model.pt"

if os.path.exists("out/ckpt.pt"):
    try:
        model.load_state_dict(torch.load("out/ckpt.pt", map_location=device))
        logger.info("✅ Resumed from checkpoint out/ckpt.pt")
        if os.path.exists("out/losses.json"):
            with open("out/losses.json", "r") as f:
                losses = json.load(f)
            start_iter = len(losses)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")

for iter in range(start_iter, max_iters):
    model.train()
    optimizer.zero_grad()
    sum_loss = 0.0

    for _ in range(accum_steps):
        xb, yb = get_batch('train')
        logits, curr_loss = model(xb, yb)
        (curr_loss / accum_steps).backward()
        sum_loss += curr_loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

    optimizer.step()
    losses.append(sum_loss)

    if iter % 100 == 0:
        logger.info(f"🔁 iter {iter}: train loss = {sum_loss:.4f}")

    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xb_val, yb_val = get_batch('val')
            _, val_loss = model(xb_val, yb_val)
        logger.info(f"✅ step {iter}: val loss = {val_loss.item():.4f}")

        # Decode sample input
        sample_ids = xb_val[0].tolist()
        decoded = tokenizer.decode(sample_ids, skip_special_tokens=True)
        logger.info(f"🧠 Sample: {decoded.strip().replace('Ġ', '')}")

        # Print state_dict keys before saving
        state_dict_keys = list(model.state_dict().keys())
        logger.info(f"State dict keys to save: {state_dict_keys} (total {len(state_dict_keys)} keys)")

        # Save checkpoint
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"📦 Checkpoint saved at {ckpt_path}")

        # Save best model if validation loss improves
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"🌟 Best model saved at {best_model_path}")

# Save final checkpoint
torch.save(model.state_dict(), ckpt_path)
logger.info(f"✅ Final model checkpoint saved at {ckpt_path}")

# Save losses
with open("out/losses.json", "w") as f:
    json.dump(losses, f)
logger.info("✅ Losses saved to out/losses.json")

# Plot losses
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Finetuning Loss')
plt.grid(True)
plt.savefig("out/finetune_loss.png")
plt.close()
logger.info("✅ Loss plot saved to out/finetune_loss.png")