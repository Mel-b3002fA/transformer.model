import sys
import os
import torch
import pickle
from transformers import AutoTokenizer
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import GPT, GPTConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 128
checkpoint_dir = 'out'
meta_path = 'out/meta.pkl'

# Load metadata (vocab size, etc.)
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

assert tokenizer.vocab_size == meta['vocab_size'], "Tokenizer vocab mismatch"

# Dynamically load the latest checkpoint
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'ckpt_*.pt'))
if not checkpoint_files:
    raise FileNotFoundError("No checkpoint files found in 'out/' directory.")
latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
checkpoint_path = latest_checkpoint
print(f"✅ Loading latest checkpoint from: {checkpoint_path}")

# Load fine-tuned model with dynamic config from checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# Handle checkpoint loading
if isinstance(checkpoint, dict):
    if any('.weight' in key or '.bias' in key for key in checkpoint.keys()):
        state_dict = checkpoint
    else:
        state_dict_keys = ['model_state_dict', 'state_dict', 'model']
        state_dict = None
        for key in state_dict_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            raise KeyError(f"Checkpoint does not contain model weights. Available keys: {list(checkpoint.keys())}")
else:
    state_dict = checkpoint

# Load model configuration and weights
model_config = GPTConfig(**checkpoint.get('config', {'vocab_size': tokenizer.vocab_size, 'block_size': block_size}))
model = GPT(model_config).to(device)
model.load_state_dict(state_dict, strict=False)
print("✅ Model loaded and ready for chat.")
model.eval()

# Text generation utilities
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

def generate(prompt, max_new_tokens=100, temperature=0.5, top_k=30, top_p=0.85,
             repetition_penalty_factor=1.2, no_repeat_ngram_size=5, debug=False, num_beams=3):
    temperature = max(temperature, 1e-5)
    model_input = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    batch_size = model_input.size(0)
    vocab_size = tokenizer.vocab_size

    # Initialize beams: (batch_idx, sequence, score)
    beams = [(0, model_input[0].clone(), 0.0)]
    finished_beams = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            new_beams = []
            for batch_idx, sequence, score in beams:
                # Check for n-gram repeats
                recent_tokens = sequence[-no_repeat_ngram_size+1:].tolist() if len(sequence) >= no_repeat_ngram_size else []
                input_crop = sequence.unsqueeze(0)[:, -block_size:]
                logits, _ = model(input_crop)
                if debug:
                    print(f"Logits shape: {logits.shape}")
                logits = logits[:, -1, :]

                logits = logits / temperature

                # Apply repetition penalty
                for token in set(sequence.tolist()):
                    logits[0, token] /= repetition_penalty_factor

                # Apply n-gram blocking
                if no_repeat_ngram_size > 0 and len(recent_tokens) >= no_repeat_ngram_size - 1:
                    ngram = tuple(recent_tokens)
                    for i in range(len(sequence) - no_repeat_ngram_size + 1):
                        match = tuple(sequence[i:i + no_repeat_ngram_size - 1].tolist())
                        if match == ngram:
                            banned_token = sequence[i + no_repeat_ngram_size - 1].item()
                            logits[0, banned_token] = -float("Inf")

                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                if debug:
                    probs = torch.softmax(logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, 5)
                    print(f"Top 5 token probs: {top_probs[0].tolist()}")
                    print(f"Top 5 token IDs: {top_indices[0].tolist()}")

                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, num_beams, dim=-1)

                for prob, token in zip(top_probs[0], top_indices[0]):
                    new_sequence = torch.cat([sequence, token.unsqueeze(0)], dim=0)
                    new_score = score + torch.log(prob).item()
                    new_beams.append((batch_idx, new_sequence, new_score))

            # Sort beams by score and keep top num_beams
            new_beams.sort(key=lambda x: x[2], reverse=True)
            beams = new_beams[:num_beams]

            # Check for finished sequences (optional: implement stopping criteria)
            # For simplicity, continue until max_new_tokens

        # Select the best beam
        best_beam = max(beams, key=lambda x: x[2])
        model_input = best_beam[1].unsqueeze(0)

    response = tokenizer.decode(model_input[0][len(tokenizer.encode(prompt)):], skip_special_tokens=True)
    return response

# Chat loop with history and parameter tuning
history = []
debug_mode = False
gen_params = {
    'temperature': 0.0,
    'top_k': 10,
    'top_p': 0.7,
    'repetition_penalty_factor': 10.0
}

try:
    print("Commands: 'exit/quit/bye' to quit, 'set param=value' to tune (e.g., 'set temp=0.7 k=40'), 'debug on/off' to toggle debug, 'history' to show past exchanges")
    while True:
        prompt = input("You: ").strip().lower()

        # Handle commands
        if prompt in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        elif prompt == 'history':
            print("\n--- Chat History ---")
            for i, (p, r) in enumerate(history[-5:], 1):
                print(f"{i}. You: {p}")
                print(f"   Joi: {r}")
            continue
        elif prompt.startswith('debug'):
            debug_mode = 'on' in prompt
            print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
            continue
        elif prompt.startswith('set'):
            parts = prompt.split()
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=')
                    try:
                        value = float(value)
                        if key in ['temp', 'temperature']:
                            gen_params['temperature'] = value
                        elif key in ['k', 'top_k']:
                            gen_params['top_k'] = int(value)
                        elif key in ['p', 'top_p']:
                            gen_params['top_p'] = value
                        elif key in ['rep', 'repetition_penalty_factor']:
                            gen_params['repetition_penalty_factor'] = value
                    except ValueError:
                        print(f"Invalid value for {key}: {value}")
            print(f"Current parameters: {gen_params}")
            continue

        # Generate response
        response = generate(
            prompt,
            max_new_tokens=100,
            temperature=gen_params['temperature'],
            top_k=gen_params['top_k'],
            top_p=gen_params['top_p'],
            repetition_penalty_factor=gen_params['repetition_penalty_factor'],
            no_repeat_ngram_size=5,
            debug=debug_mode,
            num_beams=3
        )
        print(f"Joi: {response}")

        # Store in history
        history.append((prompt, response))

except KeyboardInterrupt:
    print("\n👋 Goodbye!")