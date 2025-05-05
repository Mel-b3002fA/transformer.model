import torch
import pickle
from model import GPT, GPTConfig
from model.tokenizer import Tokenizer

# Load tokenizer metadata
with open('out/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

# Reconstruct tokenizer
tokenizer = Tokenizer()
tokenizer.stoi = meta['stoi']
tokenizer.itos = meta['itos']
tokenizer.vocab_size = meta['vocab_size']

# Load model
model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=128))
model.load_state_dict(torch.load('out/ckpt.pt', map_location='cpu'))
model.eval()

def generate_response(prompt, max_tokens=50):
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)[None, :]  # [1, T]
    for _ in range(max_tokens):
        input_ids_condensed = input_ids[:, -128:]  # crop to block size
        logits, _ = model(input_ids_condensed)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.stoi.get('.', -1):  # stop if period
            break
    return tokenizer.decode(input_ids[0].tolist())

# === Chat Loop ===
while True:
    prompt = input("You: ")
    if prompt.lower() in {"exit", "quit"}:
        break
    response = generate_response(prompt)
    print("AI:", response)
