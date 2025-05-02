from model.tokenizer import Tokenizer  # or adjust path if needed

tokenizer = Tokenizer()

# Check attributes
print("Has 'stoi'? ", hasattr(tokenizer, 'stoi'))
print("Has 'itos'? ", hasattr(tokenizer, 'itos'))

# Optional: preview a few entries
if hasattr(tokenizer, 'stoi') and hasattr(tokenizer, 'itos'):
    print("\nFirst few items in 'stoi':")
    for i, (k, v) in enumerate(tokenizer.stoi.items()):
        print(f"  {k} → {v}")
        if i >= 5:
            break

    print("\nFirst few items in 'itos':")
    for i, (k, v) in enumerate(tokenizer.itos.items()):
        print(f"  {k} → {v}")
        if i >= 5:
            break
