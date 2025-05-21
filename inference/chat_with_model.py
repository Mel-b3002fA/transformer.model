import sys
import os
import torch
import pickle
from transformers import AutoTokenizer
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='joi_chat.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to path for model import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Debug: Print Python path to verify
logging.info("Python path: %s", sys.path)

# Attempt to import from model.gpt
try:
    from model.gpt import GPT, GPTConfig, top_k_top_p_filtering
    logging.info("Successfully imported GPT, GPTConfig, top_k_top_p_filtering from model.gpt")
except ImportError as e:
    logging.error("Failed to import from model.gpt: %s", str(e))
    print(f"Error: Could not import from model/gpt.py. Ensure model/gpt.py exists and is accessible. Error: {str(e)}")
    raise

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 128  # Matches default in gpt.py; adjust if Joi uses different context length
checkpoint_dir = 'out'
meta_path = 'out/meta.pkl'

# Load metadata (vocab size, etc.)
try:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    logging.info("Metadata loaded from %s with vocab_size %d", meta_path, meta.get('vocab_size', 'unknown'))
except FileNotFoundError:
    logging.error("Metadata file not found at %s", meta_path)
    raise FileNotFoundError(f"Metadata file not found at {meta_path}")

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.vocab_size == meta['vocab_size'], f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match meta vocab size ({meta['vocab_size']})"
    logging.info("Tokenizer loaded successfully with vocab size %d", tokenizer.vocab_size)
except Exception as e:
    logging.error("Failed to load tokenizer: %s", str(e))
    raise

# Load latest checkpoint
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'ckpt_*.pt'))
if not checkpoint_files:
    logging.error("No checkpoint files found in %s", checkpoint_dir)
    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
checkpoint_path = latest_checkpoint
logging.info("Loading latest checkpoint from: %s", checkpoint_path)

# Load checkpoint
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
except Exception as e:
    logging.error("Failed to load checkpoint %s: %s", checkpoint_path, str(e))
    raise

# Handle checkpoint loading
if isinstance(checkpoint, dict):
    state_dict_keys = ['model_state_dict', 'state_dict', 'model']
    state_dict = None
    for key in state_dict_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None:
        logging.error("Checkpoint does not contain model weights. Available keys: %s", list(checkpoint.keys()))
        raise KeyError(f"Checkpoint does not contain model weights. Available keys: {list(checkpoint.keys())}")
else:
    state_dict = checkpoint

# Load model configuration and weights
try:
    config_dict = checkpoint.get('config', {
        'vocab_size': tokenizer.vocab_size,
        'block_size': block_size,
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 256
    })
    model_config = GPTConfig(**config_dict)
    model = GPT(model_config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logging.info("Model loaded successfully with config: %s", config_dict)
except Exception as e:
    logging.error("Failed to load model: %s", str(e))
    raise

# Chat loop with history and parameter tuning
history = []
debug_mode = False
gen_params = {
    'temperature': 0.7,  # Adjusted for more diverse outputs
    'top_k': 50,
    'top_p': 0.9,
    'repetition_penalty_factor': 1.2
}

print("Commands: 'exit/quit/bye' to quit, 'set param=value' to tune (e.g., 'set temp=0.7 k=50 p=0.9 rep=1.2'), 'debug on/off' to toggle debug, 'history' to show past exchanges")
try:
    while True:
        prompt = input("You: ").strip()

        # Handle commands
        if prompt.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            logging.info("Chat session terminated by user")
            break
        elif prompt.lower() == 'history':
            print("\n--- Chat History ---")
            for i, (p, r) in enumerate(history[-5:], 1):
                print(f"{i}. You: {p}")
                print(f"   Joi: {r}")
            continue
        elif prompt.lower().startswith('debug'):
            debug_mode = 'on' in prompt.lower()
            print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
            logging.info("Debug mode set to %s", debug_mode)
            continue
        elif prompt.lower().startswith('set'):
            parts = prompt.split()
            for part in parts[1:]:
                if '=' in part:
                    key, value = part.split('=')
                    try:
                        value = float(value)
                        if key in ['temp', 'temperature']:
                            gen_params['temperature'] = value
                        elif key in [' planting', 'top_k']:
                            gen_params['top_k'] = int(value)
                        elif key in ['p', 'top_p']:
                            gen_params['top_p'] = value
                        elif key in ['rep', 'repetition_penalty_factor']:
                            gen_params['repetition_penalty_factor'] = value
                        logging.info("Set parameter %s=%s", key, value)
                    except ValueError:
                        print(f"Invalid value for {key}: {value}")
                        logging.warning("Invalid parameter value: %s=%s", key, value)
            print(f"Current parameters: {gen_params}")
            continue

        # Build context with history
        context = ""
        for past_prompt, past_response in history[-5:]:  # Limit to last 5 exchanges
            context += f"User: {past_prompt}\nJoi: {past_response}\n"
        context += f"User: {prompt}\nJoi: "
        input_ids = tokenizer(context, return_tensors='pt').input_ids.to(device)

        # Generate response
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    temperature=gen_params['temperature'],
                    top_k=gen_params['top_k'],
                    top_p=gen_params['top_p'],
                    repetition_penalty_factor=gen_params['repetition_penalty_factor'],
                    no_repeat_ngram_size=5
                )
            response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
            print(f"Joi: {response}")
            logging.info("Generated response for prompt '%s': %s", prompt, response)

            # Store in history
            history.append((prompt, response))

            # Debug output
            if debug_mode:
                print(f"Input IDs shape: {input_ids.shape}")
                print(f"Output IDs shape: {output_ids.shape}")
                logging.debug("Input IDs shape: %s, Output IDs shape: %s", input_ids.shape, output_ids.shape)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            logging.error("Generation error: %s", str(e))

except KeyboardInterrupt:
    print("\n👋 Goodbye!")
    logging.info("Chat session interrupted by user")