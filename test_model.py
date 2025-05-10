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
