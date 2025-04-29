d_model = 512
num_heads = 8
num_layers = 6
ff_dim = 2048
dropout_rate = 0.1
max_seq_len = 128
vocab_size = 5000
batch_size = 32


config = {

    "vocab_size": 10000,         
    "block_size": 128,           
    "n_layer": 6,                
    "n_head": 8,                 
    "n_embd": 512,               

 
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "max_iters": 5000,
    "eval_interval": 500,
    "eval_iters": 200,

    # Generation settings
    "temperature": 1.0,
    "top_k": 50,

    # Save/load
    "checkpoint_path": "assets/best_model.pt",
    "tokenizer_path": "assets/tokenizer.json",
}
