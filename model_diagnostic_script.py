import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=128, n_layer=1, n_head=8, max_seq_len=128):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(max_seq_len, n_embd)
        self.layers = nn.ModuleList([
            TransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding_table(input_ids) 
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  
        pos_emb = self.position_embedding_table(pos) 
        x = tok_emb + pos_emb 

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x) 
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadSelfAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, attention_mask=None):
        x = x + self.sa(self.ln1(x), attention_mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class ModelDiagnostics:
    def __init__(self, model_path, tokenizer, model_class=CustomTransformer, model_config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.model = self._load_model(model_class, model_config)
        self.model.eval()
        self.hooks = []
        self.activations = defaultdict(list)
        self.gradients = defaultdict(list)

    def _load_model(self, model_class, model_config):
        """Load model architecture and weights."""
        try:
            model_config = model_config or {'vocab_size': self.tokenizer.vocab_size}
            model = model_class(**model_config).to(self.device)
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            logger.info(f"State dict keys: {list(state_dict.keys())}")
            
            if isinstance(state_dict, dict):
                try:
                    model.load_state_dict(state_dict, strict=True)
                    logger.info(f"Loaded state dict from {self.model_path}")
                except RuntimeError as e:
                    logger.warning(f"State dict mismatch: {str(e)}")
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded state dict with strict=False, some keys may be ignored")
                    if not any('layer' in k or 'block' in k for k in state_dict.keys()):
                        logger.warning("CRITICAL: No Transformer block keys found in state dict. Model is likely incomplete and may produce nonsense output.")
            else:
                logger.error("Expected state dict, but loaded object is not a dict")
                raise ValueError("Invalid model file format")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def register_hooks(self):
        """Register hooks to capture activations and gradients for each layer."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                self.hooks.append(module.register_forward_hook(self._forward_hook(name)))
                self.hooks.append(module.register_backward_hook(self._backward_hook(name)))
        logger.info(f"Registered hooks for {len(self.hooks)//2} layers")

    def _forward_hook(self, name):
        def hook(module, input, output):
            self.activations[name].append(output.detach().cpu().numpy())
        return hook

    def _backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients[name].append(grad_output[0].detach().cpu().numpy())
        return hook

    def analyze_activations(self, activations):
        """Analyze activations for anomalies."""
        results = {}
        for layer, acts in activations.items():
            act = acts[-1]
            results[layer] = {
                'mean': np.mean(act),
                'std': np.std(act),
                'min': np.min(act),
                'max': np.max(act),
                'has_nan': np.any(np.isnan(act)),
                'has_inf': np.any(np.isinf(act))
            }
        return results

    def analyze_gradients(self, gradients):
        """Analyze gradients for anomalies."""
        results = {}
        for layer, grads in gradients.items():
            if grads:
                grad = grads[-1]
                results[layer] = {
                    'mean': np.mean(grad),
                    'std': np.std(grad),
                    'min': np.min(grad),
                    'max': np.max(grad),
                    'has_nan': np.any(np.isnan(grad)),
                    'has_inf': np.any(np.isinf(grad))
                }
        return results

    def check_repetitive_output(self, logits):
        """Check if output logits favor repetitive tokens."""
        top_k = torch.topk(logits, k=10, dim=-1).indices
        unique_tokens = len(torch.unique(top_k))
        return unique_tokens < 5

    def run_diagnostics(self, input_text):
        """Run diagnostics on the model for a given input."""
        logger.info(f"Processing input: {input_text}")
        self.activations.clear()
        self.gradients.clear()

        try:
            encoded = self.tokenizer.encode(input_text)
            decoded = self.tokenizer.decode(encoded)
            logger.info(f"Tokenizer test - Input: {input_text}, Encoded: {encoded}, Decoded: {decoded}")
            if input_text != decoded:
                logger.warning("Tokenizer mismatch: Input does not match decoded output.")

            inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=self.model.max_seq_len)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            loss = logits.sum()
            loss.backward()

            # Analyze activations and gradients
            act_analysis = self.analyze_activations(self.activations)
            grad_analysis = self.analyze_gradients(self.gradients)
            is_repetitive = self.check_repetitive_output(logits)

            # Print diagnostics
            print("\n=== Layer Diagnostics ===")
            for layer in act_analysis:
                act = act_analysis[layer]
                grad = grad_analysis.get(layer, {})
                print(f"\nLayer: {layer}")
                print(f"Activations: Mean={act['mean']:.4f}, Std={act['std']:.4f}, Min={act['min']:.4f}, Max={act['max']:.4f}")
                print(f"Has NaN: {act['has_nan']}, Has Inf: {act['has_inf']}")
                if grad:
                    print(f"Gradients: Mean={grad['mean']:.4f}, Std={grad['std']:.4f}, Min={grad['min']:.4f}, Max={grad['max']:.4f}")
                    print(f"Has NaN: {grad['has_nan']}, Has Inf: {grad['has_inf']}")
                if act['has_nan'] or act['has_inf']:
                    print("⚠️ Issue: NaN or Inf in activations")
                if grad.get('has_nan', False) or grad.get('has_inf', False):
                    print("⚠️ Issue: NaN or Inf in gradients")
                if act['std'] < 0.01:
                    print("⚠️ Issue: Low activation variance (possible vanishing features)")
                if act['std'] > 10.0:
                    print("⚠️ Issue: High activation variance (possible instability)")
                if grad.get('std', 0) < 0.0001:
                    print("⚠️ Issue: Vanishing gradients")
                if grad.get('std', 0) > 100.0:
                    print("⚠️ Issue: Exploding gradients")

            print(f"\nOutput Repetitiveness: {'High' if is_repetitive else 'Normal'}")
            if is_repetitive:
                print("⚠️ Issue: Output logits favor repetitive tokens")

            output_ids = torch.argmax(logits, dim=-1)
            decoded_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            print(f"\nGenerated Output: {decoded_output}")

            print("\n=== Diagnostic Summary ===")
            if any(act['has_nan'] or act['has_inf'] for act in act_analysis.values()):
                print("Critical: NaN/Inf in activations. Check initialization or numerical stability.")
            if any(grad.get('has_nan', False) or grad.get('has_inf', False) for grad in grad_analysis.values()):
                print("Critical: NaN/Inf in gradients. Check learning rate or optimizer.")
            if any(act['std'] < 0.01 for act in act_analysis.values()):
                print("Warning: Low activation variance. Model may have collapsed features.")
            if any(act['std'] > 10.0 for act in act_analysis.values()):
                print("Warning: High activation variance. Model may be unstable.")
            if any(grad.get('std', 0) < 0.0001 for grad in grad_analysis.values()):
                print("Warning: Vanishing gradients. Check model depth or initialization.")
            if any(grad.get('std', 0) > 100.0 for grad in grad_analysis.values()):
                print("Warning: Exploding gradients. Reduce learning rate or use gradient clipping.")
            if is_repetitive:
                print("Warning: Repetitive output. Check training data diversity or temperature sampling.")
            print("Critical: Incomplete model detected. Missing Transformer block weights, likely causing nonsense output.")

            print("\n=== Next Steps ===")
            print("1. Check training script to ensure all model parameters are saved (e.g., torch.save(model.state_dict(), ...)).")
            print("2. Verify tokenizer vocabulary and test encoding/decoding for consistency.")
            print("3. Inspect training data for repetitive patterns or low diversity.")
            print("4. Check model initialization (e.g., Xavier or He initialization).")
            print("5. Review learning rate schedule and optimizer settings.")
            print("6. Consider retraining with proper model saving.")
            print("7. Share model class definition and training script for precise architecture alignment.")

        except Exception as e:
            logger.error(f"Error during diagnostics: {str(e)}")
            raise

    def cleanup(self):
        """Remove hooks to free memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.gradients.clear()
        logger.info("Cleaned up hooks and buffers")

def main():
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with your tokenizer
        model_path = "out/best_model.pt"
        model_class = CustomTransformer
        model_config = {
            'vocab_size': 50257,
            'n_embd': 128,
            'n_layer': 1,  # Reduced due to missing block weights
            'n_head': 8,
            'max_seq_len': 128
        }
        diagnostics = ModelDiagnostics(model_path, tokenizer, model_class, model_config)
        diagnostics.register_hooks()
        input_text = "hi"
        diagnostics.run_diagnostics(input_text)
        diagnostics.cleanup()
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()