import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import GPT, GPTConfig
from model.tokenizer import Tokenizer
import torch
import pickle

# === Config ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 128  # same as during training
checkpoint_path = 'out/ckpt.pt'
meta_path = 'out/meta.pkl'

# === Load tokenizer metadata ===
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

tokenizer = Tokenizer()
# Load tokenizer vocab from meta.pkl
with open("out/meta.pkl", "rb") as f:
    meta = pickle.load(f)
tokenizer.stoi = meta['stoi']
tokenizer.itos = meta['itos']
tokenizer.vocab_size = meta['vocab_size']

# === Load model ===
model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size)).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("✅ Model loaded and ready for chat.")

# === Generate text ===
def generate(prompt, max_new_tokens=50):
    model_input = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)[None, :].to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_crop = model_input[:, -block_size:]
            logits, _ = model(input_crop)
            logits = logits[:, -1, :]  # last token's logits
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            model_input = torch.cat((model_input, next_token), dim=1)

    output_text = tokenizer.decode(model_input[0].tolist())
    return output_text

# === Chat loop ===
print("💬 Type your prompt below (or type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    output = generate(user_input, max_new_tokens=100)
    print("AI:", output[len(user_input):].strip())  # Show only the response
