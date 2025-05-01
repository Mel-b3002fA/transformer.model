import torch
import pickle
from model import GPT, GPTConfig



with open('data/openwebtext/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

def encode(s):
    return [stoi.get(c, stoi['<unk>']) for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

checkpoint = torch.load('out/ckpt.pt', map_location='cpu')
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(checkpoint['model'])
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

def generate(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    device = next(model.parameters()).device
    model.eval()

    idx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

    return decode(idx[0].tolist())

# --- Conversation history buffer
conversation = ""

# Chat loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    conversation += f"You: {user_input}\nBot:"

    response = generate(model, conversation, max_new_tokens=100)

    response = response.split('\nYou:')[0].strip()

    print(f"Bot: {response}")

    conversation += f" {response}\n"


