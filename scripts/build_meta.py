""" import os
import pickle
from model.tokenizers import Tokenizer  # Adjust if your tokenizer class is named differently


tokenizer = Tokenizer()


itos = tokenizer.get_itos() 
stoi = tokenizer.get_stoi()  

meta = {
    'vocab_size': len(itos),
    'itos': itos,
    'stoi': stoi
}

os.makedirs('data', exist_ok=True)

with open('data/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("meta.pkl saved to data/meta.pkl")
 """


import pickle
from model.tokenizer import Tokenizer  # adjust if necessary

tokenizer = Tokenizer()
meta = {
    'vocab_size': len(tokenizer.vocab),
    'itos': tokenizer.itos,  # index to string
    'stoi': tokenizer.stoi,  # string to index
}

with open('data/openwebtext/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
