import sys
import os
import torch
import pickle
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import GPT, GPTConfig

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 128
checkpoint_path = 'out/ckpt.pt'
meta_path = 'out/meta.pkl'

# Load meta information (optional check)
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

assert tokenizer.vocab_size == meta['vocab_size'], "Tokenizer vocab mismatch"

model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size)).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
model.eval()
print("✅ Model loaded and ready for chat.")


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    logits = logits.clone()
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < min_values, filter_value, logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        for batch_idx in range(logits.size(0)):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = filter_value

    return logits

def repetition_penalty(logits, past_tokens, penalty=1.0):
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for token in past_tokens:
        logits[0, token] /= penalty
    return logits

def no_repeat_ngram_filtering(logits, model_input, no_repeat_ngram_size=0):
    if no_repeat_ngram_size < 1:
        return logits
    logits = logits.clone()
    input_ids = model_input[0].tolist()
    if len(input_ids) < no_repeat_ngram_size:
        return logits

    ngrams = {}
    for i in range(len(input_ids) - no_repeat_ngram_size + 1):
        ngram = tuple(input_ids[i:i + no_repeat_ngram_size - 1])
        next_token = input_ids[i + no_repeat_ngram_size - 1]
        if ngram not in ngrams:
            ngrams[ngram] = []
        ngrams[ngram].append(next_token)

    current_ngram = tuple(input_ids[-(no_repeat_ngram_size - 1):])
    if current_ngram in ngrams:
        banned_tokens = ngrams[current_ngram]
        for token in banned_tokens:
            logits[0, token] = -float('Inf')
    return logits

def generate(prompt, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.95,
             repetition_penalty_factor=1.2, no_repeat_ngram_size=3):
    temperature = max(temperature, 1e-5)
    model_input = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    recent_tokens = model_input[0].tolist()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_crop = model_input[:, -block_size:]
            logits, _ = model(input_crop)
            logits = logits[:, -1, :]  


            logits = logits / temperature

            for token in set(recent_tokens):
                logits[0, token] /= repetition_penalty_factor

            if no_repeat_ngram_size > 0 and len(recent_tokens) >= no_repeat_ngram_size:
                ngram = tuple(recent_tokens[-(no_repeat_ngram_size - 1):])
                for i in range(len(recent_tokens) - no_repeat_ngram_size + 1):
                    match = tuple(recent_tokens[i:i + no_repeat_ngram_size - 1])
                    if match == ngram:
                        banned_token = recent_tokens[i + no_repeat_ngram_size - 1]
                        logits[0, banned_token] = -float("Inf")

            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            model_input = torch.cat([model_input, next_token], dim=1)
            recent_tokens.append(next_token.item())

    response = tokenizer.decode(model_input[0][len(tokenizer.encode(prompt)):], skip_special_tokens=True)
    return response



try:
    while True:
        prompt = input("You: ")
        if prompt.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        response = generate(prompt)
        print(f"Joi: {response}")
except KeyboardInterrupt:
    print("\n👋 Goodbye!")
