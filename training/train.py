import sys
import os
import torch
import pickle
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
import glob
import logging
from datetime import datetime

# Add parent directory to path for model import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Debug: Log Python path
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='train_log.txt')
logger = logging.getLogger(__name__)
logger.info("Python path: %s", sys.path)

# Import model
try:
    from model.gpt import GPT, GPTConfig
    logger.info("Successfully imported GPT, GPTConfig from model.gpt")
except ImportError as e:
    logger.error("Failed to import from model.gpt: %s", str(e))
    raise

# Hyperparameters
batch_size = 4
block_size = 128  # Matches gpt.py; adjust if dataset requires longer context
max_iters = 10_000
learning_rate = 1e-4
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gradient_clip = 1.0
accum_steps = 8

# Initialize tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded with vocab size %d", tokenizer.vocab_size)
except Exception as e:
    logger.error("Failed to load tokenizer: %s", str(e))
    raise

# Create output directory
os.makedirs('out', exist_ok=True)

# Save vocabulary metadata
vocab = tokenizer.get_vocab()
stoi = vocab
itos = {idx: token for token, idx in stoi.items()}
meta_data = {'vocab_size': tokenizer.vocab_size, 'stoi': stoi, 'itos': itos, 'block_size': block_size}
with open('out/meta.pkl', 'wb') as f:
    pickle.dump(meta_data, f)
logger.info("✅ meta.pkl successfully saved with vocab_size=%d, block_size=%d", tokenizer.vocab_size, block_size)

# Load and preprocess dataset
try:
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    logger.info("Loaded Alpaca dataset with %d samples", len(dataset))
except Exception as e:
    logger.error("Failed to load dataset: %s", str(e))
    raise

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
logger.info(f"✅ Loaded {len(train_data)} training and {len(val_data)} validation samples")

# Log sample data for debugging
sample_data = tokenized_data[0][:20]
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
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=6,  # Matches gpt.py default
    n_head=6,   # Matches gpt.py default
    n_embd=256  # Matches gpt.py default
)
model = GPT(config).to(device)
logger.info("✅ Model initialized with config: vocab_size=%d, block_size=%d, n_layer=%d, n_head=%d, n_embd=%d",
            config.vocab_size, config.block_size, config.n_layer, config.n_head, config.n_embd)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Initialize lists for tracking losses
train_losses = []
val_losses = []
start_iter = 0
best_val_loss = float('inf')
ckpt_path = f"out/ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
best_model_path = "out/best_model.pt"

# Load latest checkpoint if exists
checkpoint_files = glob.glob(os.path.join('out', 'ckpt_*.pt'))
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    try:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        logger.info("✅ Resumed from checkpoint %s", latest_checkpoint)
        if os.path.exists("out/losses.json"):
            with open("out/losses.json", "r") as f:
                saved_losses = json.load(f)
                train_losses = saved_losses.get('train_losses', [])
                val_losses = saved_losses.get('val_losses', [])
            start_iter = len(train_losses)
    except Exception as e:
        logger.warning(f"Failed to load checkpoint %s: %s", latest_checkpoint, str(e))

# Training loop
for iter in range(start_iter, max_iters):
    model.train()
    optimizer.zero_grad()
    sum_loss = 0.0

    for _ in range(accum_steps):
        xb, yb = get_batch('train')
        logits, curr_loss = model(xb, yb)
        (curr_loss / accum_steps).backward()
        sum_loss += curr_loss.item()

    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

    optimizer.step()
    train_losses.append(sum_loss)

    if iter % 100 == 0:
        logger.info(f"🔁 iter {iter}: train loss = {sum_loss:.4f}")

    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xb_val, yb_val = get_batch('val')
            _, val_loss = model(xb_val, yb_val)
        val_losses.append(val_loss.item())
        logger.info(f"✅ step {iter}: val loss = {val_loss.item():.4f}")

        # Decode sample input
        sample_ids = xb_val[0].tolist()
        decoded = tokenizer.decode(sample_ids, skip_special_tokens=True)
        logger.info(f"🧠 Sample: {decoded.strip().replace('Ġ', '')}")

        # Save checkpoint with config
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': config.vocab_size,
                'block_size': config.block_size,
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_embd': config.n_embd
            }
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"📦 Checkpoint saved at {ckpt_path}")

        # Save best model if validation loss improves
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(checkpoint, best_model_path)
            logger.info(f"🌟 Best model saved at {best_model_path}")

# Save final checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': config.vocab_size,
        'block_size': config.block_size,
        'n_layer': config.n_layer,
        'n_head': config.n_head,
        'n_embd': config.n_embd
    }
}
torch.save(checkpoint, ckpt_path)
logger.info(f"✅ Final model checkpoint saved at {ckpt_path}")

# Save losses
with open("out/losses.json", "w") as f:
    json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)
logger.info("✅ Losses saved to out/losses.json")

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(range(0, len(val_losses) * eval_interval, eval_interval), val_losses, label='Validation Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("out/finetune_loss.png")
plt.close()
logger.info("✅ Loss plot saved to out/finetune_loss.png")