""" import os
import time
import math
import pickle
from contextlib import nullcontext
import matplotlib.pyplot as plt
from IPython.display import clear_output
 
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from datasets import load_dataset
from torch.utils.data import DataLoader

from model import GPTConfig, GPT

out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False 
always_save_checkpoint = True
init_from = 'scratch'

wandb_log = False 
wandb_project = 'owt'
wandb_run_name = 'gpt2'

dataset = 'the-pile'  # Updated to reflect actual dataset
gradient_accumulation_steps = 5 * 8 
batch_size = 12 
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 
bias = False 

learning_rate = 6e-4 
max_iters = 600000 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 

decay_lr = True 
warmup_iters = 2000 
lr_decay_iters = 600000 
min_lr = 6e-5 
backend = 'nccl' 

device = 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True 

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) 
config = {k: globals()[k] for k in config_keys} 


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
    seed_offset = ddp_rank 
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu' 

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load The Pile dataset
print("Loading The Pile dataset...")
pile_dataset = load_dataset("monology/pile-10k", split="train")

# Use GPT-2 tokenizer from Hugging Face if needed
from transformers import GPT2Tokenizer
hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_tokenizer.pad_token = hf_tokenizer.eos_token

def encode_pile(example):
    tokens = hf_tokenizer.encode(example['text'], truncation=True, max_length=block_size+1)
    return {"input_ids": tokens[:-1], "labels": tokens[1:]}

pile_dataset = pile_dataset.map(encode_pile, remove_columns=["text"])
pile_dataset.set_format(type='torch', columns=["input_ids", "labels"])

pile_loader = DataLoader(pile_dataset, batch_size=batch_size, shuffle=True)

def get_batch():
    batch = next(iter(pile_loader))
    x = batch["input_ids"].to(device)
    y = batch["labels"].to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    print("Initializing a new model from thin air")
    model_args['vocab_size'] = hf_tokenizer.vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None 

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) 

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y = get_batch() 
t0 = time.time()
local_iter_num = 0 
raw_model = model.module if ddp else model 
running_mfu = -1.0

while iter_num < max_iters:
    lr = get_lr(iter_num)
    optimizer.param_groups[0]['lr'] = lr

    X, Y = get_batch()

    with ctx:
        logits, loss = model(X, Y)
    loss = loss / gradient_accumulation_steps
    loss.backward()

    if iter_num % gradient_accumulation_steps == 0:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    if iter_num % log_interval == 0 and master_process:
        print(f"Iteration {iter_num}: Loss = {loss.item():.4f}, LR = {lr:.6f}")
        if wandb_log:
            wandb.log({'loss': loss.item(), 'lr': lr, 'iter': iter_num})

    if iter_num % eval_interval == 0 and master_process:
        print("Evaluating model...")
        losses = estimate_loss()
        print(f"Training loss: {losses['train']:.4f}, Validation loss: {losses['val']:.4f}")

    if iter_num % eval_interval == 0 and master_process and always_save_checkpoint:
        print("Saving checkpoint...")
        checkpoint = {
            'iter_num': iter_num,
            'model_args': model_args,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    iter_num += 1 """



import sys
import os
import torch
import pickle
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig
from model.tokenizer import Tokenizer

# === Config ===
batch_size = 4
block_size = 128
max_iters = 200
learning_rate = 1e-3
eval_interval = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load data ===
with open('data/joi.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# === Tokenizer ===
tokenizer = Tokenizer()
tokenizer.train(text)
vocab_size = tokenizer.vocab_size

# Save metadata
os.makedirs('out', exist_ok=True)
with open('out/meta.pkl', 'wb') as f:
    pickle.dump({
        'vocab_size': tokenizer.vocab_size,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos
    }, f)
print("✅ meta.pkl successfully saved.")

# === Encode dataset ===
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
train_data = data[:int(0.9 * len(data))]
val_data = data[int(0.9 * len(data)):]

# === Batch function ===
def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i + block_size] for i in ix])
    y = torch.stack([data_split[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# === Model ===
model = GPT(GPTConfig(vocab_size=vocab_size, block_size=block_size)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# === Training Loop ===
losses = []

for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if iter % eval_interval == 0:
        print(f"step {iter}: loss = {loss.item():.4f}")

os.makedirs("out", exist_ok=True)
torch.save(model.state_dict(), "out/ckpt.pt")
print("✅ Model checkpoint saved at out/ckpt.pt")

with open("out/losses.json", "w") as f:
    json.dump(losses, f)
print("✅ Losses saved to out/losses.json")



