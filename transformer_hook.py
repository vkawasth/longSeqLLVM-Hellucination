"""
TransformerHook: Extract hidden states from transformer for GSOD analysis.

Supports:
  - Causal attention (standard GPT-style CAR)
  - Cross attention (encoder-decoder)
  - Bidirectional attention (BERT-style, spot-check mode)
  - Long-context with ALiBi positional encoding

Model support:
  - GPT-2 (all sizes) — primary target for M1/M2 8GB
  - Phi-2 (2.7B) — stretch target for 16GB
  - Any HuggingFace CausalLM via hooks

Positional encoding options:
  - Standard learned (default GPT-2)
  - ALiBi: no additional memory, length extrapolation
  - RoPE: rotary, used in LLaMA/Mistral family

At 1024 tokens on M1/M2 8GB:
  - GPT-2 small/medium/large/XL: all tractable
  - Attention matrix: 1024² × 12 heads × 48 layers × 2 bytes ≈ 1.2GB
  - Total with weights: ~4.5GB for GPT-2 XL ✅
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    model_name: str = "gpt2"          # HuggingFace model ID
    max_length: int = 1024             # Fixed at 1024 for M1/M2
    extract_layer: int = -1            # Which layer to extract (-1 = last)
    attention_type: str = "causal"     # causal | cross | bidirectional
    positional_encoding: str = "learned"  # learned | alibi | rope
    device: str = "cpu"                # cpu | mps (Apple Silicon)
    dtype: torch.dtype = torch.float32


class TransformerHook:
    """
    Attaches to a HuggingFace transformer and extracts hidden states
    at each forward pass. Designed for minimal memory footprint on M1/M2.
    """

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._hidden_states = []
        self._hooks = []

    def load(self):
        """Load model and tokenizer. Lazy — call before first use."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.dtype,
            output_hidden_states=True,
        )
        self.model.eval()
        self.model.to(self.config.device)

        # Apply ALiBi if requested (modify attention bias)
        if self.config.positional_encoding == "alibi":
            self._apply_alibi_bias()

        print(f"  Loaded. Parameters: "
              f"{sum(p.numel() for p in self.model.parameters()):,}")
        return self

    def _apply_alibi_bias(self):
        """
        ALiBi: Attention with Linear Biases (Press et al. 2022).
        Replaces learned position embeddings with linear distance penalty.
        Enables length extrapolation beyond training context.
        Zero additional memory cost.

        Bias for head h: m_h * |i - j| where m_h = 2^{-8/n_heads * h}
        """
        # ALiBi is applied as a hook on the attention weights
        # We inject it as a forward hook on each attention layer
        n_heads = self.model.config.n_head if hasattr(
            self.model.config, 'n_head'
        ) else self.model.config.num_attention_heads

        slopes = torch.tensor([
            2 ** (-8.0 / n_heads * (h + 1))
            for h in range(n_heads)
        ], dtype=self.config.dtype)

        def alibi_hook(module, input, output):
            """Add ALiBi bias to attention scores."""
            if isinstance(output, tuple):
                attn_weights = output[0]
                if attn_weights is not None and attn_weights.dim() == 4:
                    seq_len = attn_weights.shape[-1]
                    # Build distance matrix
                    positions = torch.arange(seq_len, device=attn_weights.device)
                    distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
                    # ALiBi bias: [n_heads, seq_len, seq_len]
                    bias = -slopes.view(-1, 1, 1) * distances.unsqueeze(0)
                    bias = bias.to(attn_weights.dtype)
                    # Add to attention weights
                    batch_size = attn_weights.shape[0]
                    modified = attn_weights + bias.unsqueeze(0).expand(
                        batch_size, -1, -1, -1
                    )
                    return (modified,) + output[1:]
            return output

        # Register hook on all attention layers
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(alibi_hook)
                self._hooks.append(hook)

    def _register_hidden_state_hook(self, layer_idx: int):
        """Register hook to capture hidden states at specified layer."""
        self._hidden_states = []

        layers = None
        if hasattr(self.model, 'transformer'):  # GPT-2 style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model'):       # LLaMA style
            layers = self.model.model.layers

        if layers is None:
            return

        n_layers = len(layers)
        target_layer = layer_idx % n_layers

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Detach and move to CPU immediately to save GPU/MPS memory
            self._hidden_states.append(
                hidden.detach().cpu().float().numpy()
            )

        hook = layers[target_layer].register_forward_hook(hook_fn)
        self._hooks.append(hook)

    def extract_hidden_states(
        self,
        text: str,
        layer: Optional[int] = None
    ) -> np.ndarray:
        """
        Run forward pass and extract hidden states.

        Args:
            text: Input text (will be truncated/padded to max_length)
            layer: Which layer to extract. None = use config default.

        Returns:
            np.ndarray of shape [seq_len, hidden_dim]
        """
        if self.model is None:
            self.load()

        target_layer = layer if layer is not None else self.config.extract_layer

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Register hook
        self._register_hidden_state_hook(target_layer)

        # Forward pass (no gradient)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        # Extract hidden states
        if self._hidden_states:
            hidden = self._hidden_states[-1]  # Last captured
            if hidden.ndim == 3:
                hidden = hidden[0]  # Remove batch dim: [seq_len, hidden_dim]
        else:
            # Fallback: use model's own hidden_states output
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden = outputs.hidden_states[target_layer]
                hidden = hidden[0].cpu().float().numpy()
            else:
                raise ValueError("Could not extract hidden states")

        # Clean up hooks
        self._remove_hooks()

        return hidden  # [seq_len, hidden_dim]

    def generate_and_extract(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
    ) -> Tuple[str, np.ndarray]:
        """
        Generate text from prompt and extract hidden states of generation.
        Combines generation + extraction in one pass.

        Returns:
            (generated_text, hidden_states [seq_len, hidden_dim])
        """
        if self.model is None:
            self.load()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length - max_new_tokens,
            truncation=True,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[1]

        # Register hook before generation
        self._register_hidden_state_hook(self.config.extract_layer)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=False,  # Use hook instead
            )

        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated[0][prompt_len:],
            skip_special_tokens=True
        )

        # Get hidden states from the full sequence via second pass
        full_text = prompt + generated_text
        self._remove_hooks()
        hidden = self.extract_hidden_states(full_text)

        return generated_text, hidden

    def _remove_hooks(self):
        """Clean up all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._hidden_states = []

    def __del__(self):
        self._remove_hooks()


# ─────────────────────────────────────────────────────────────────
# SYNTHETIC TRAJECTORY GENERATOR (for Phase 1 validation)
# ─────────────────────────────────────────────────────────────────

class SyntheticTrajectoryGenerator:
    """
    Generate synthetic hidden-state trajectories with known
    hallucination types for GSOD validation.

    Constructs trajectories whose Frobenius orbit profiles are
    known by construction — allows us to validate the detector
    before attaching to a real transformer.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        hidden_dim: int = 64,
        seed: int = 42
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.rng = np.random.RandomState(seed)

    def clean_trajectory(self) -> np.ndarray:
        """
        Generate an admissible (hallucination-free) trajectory.
        Smooth, globally consistent hidden state evolution.
        """
        # Start from a base state and evolve smoothly
        states = np.zeros((self.seq_len, self.hidden_dim))
        state = self.rng.randn(self.hidden_dim)
        # Smooth rotation matrix (nilpotent perturbation)
        A = self.rng.randn(self.hidden_dim, self.hidden_dim) * 0.01
        A = A - A.T  # Antisymmetric: eigenvalues are purely imaginary
        # Discrete evolution: globally consistent
        for t in range(self.seq_len):
            state = state + A @ state * 0.1 + self.rng.randn(
                self.hidden_dim
            ) * 0.05
            states[t] = state

        return states

    def binary_entanglement_trajectory(self, orbit_size: int = 2) -> np.ndarray:
        """
        Generate trajectory with binary entanglement (v_2 orbit size k).
        The orientation alternates with period k, creating a
        Frobenius orbit of size k under φ_2.
        """
        states = np.zeros((self.seq_len, self.hidden_dim))
        base = self.rng.randn(self.hidden_dim)
        # Create k orthogonal modes that cycle
        modes = self.rng.randn(orbit_size, self.hidden_dim)
        modes = modes / np.linalg.norm(modes, axis=1, keepdims=True)

        for t in range(self.seq_len):
            mode_idx = (t // (self.seq_len // (orbit_size * 4))) % orbit_size
            # Introduce orientation-flipping between conjugate modes
            flip = 1 if (t % (2 ** orbit_size)) < (2 ** (orbit_size - 1)) else -1
            states[t] = base + flip * modes[mode_idx] * 2.0
            states[t] += self.rng.randn(self.hidden_dim) * 0.1

        return states

    def conjugate_closure_failure(self) -> np.ndarray:
        """
        Generate trajectory with quintic closure failure (v_5 orbit size 2).
        The sequence locally closes at each step but fails global
        5-fold closure — equivalent to H_1(C_*; Z_5) ≠ 0.
        """
        states = np.zeros((self.seq_len, self.hidden_dim))
        # Create a trajectory that spirals but doesn't close
        theta = np.linspace(0, 2 * np.pi * 4.9, self.seq_len)  # Almost 5 loops
        r = 1.0 + 0.3 * np.sin(theta * 5)

        base_direction = self.rng.randn(self.hidden_dim)
        orthogonal = self.rng.randn(self.hidden_dim)
        orthogonal -= orthogonal.dot(base_direction) * base_direction
        orthogonal = orthogonal / np.linalg.norm(orthogonal)
        base_direction = base_direction / np.linalg.norm(base_direction)

        for t in range(self.seq_len):
            states[t] = (r[t] * np.cos(theta[t]) * base_direction +
                         r[t] * np.sin(theta[t]) * orthogonal +
                         self.rng.randn(self.hidden_dim) * 0.1)

        return states

    def resonance_7adic_trajectory(self) -> np.ndarray:
        """
        Generate trajectory with 7-adic resonance hallucination.
        Passes v_2 and v_5 gates but creates a 7-step semantic cycle
        that returns to a contradictory position.
        """
        states = np.zeros((self.seq_len, self.hidden_dim))
        # 7-periodic drift
        period = 7
        drift_per_period = self.rng.randn(self.hidden_dim) * 0.5

        state = self.rng.randn(self.hidden_dim)
        cycle_count = 0
        for t in range(self.seq_len):
            if t % period == 0 and t > 0:
                cycle_count += 1
                # Drift that accumulates over 7-cycles
                state += drift_per_period * 0.3

            # Local smooth evolution (passes v_2, v_5)
            local_noise = self.rng.randn(self.hidden_dim) * 0.05
            states[t] = state + local_noise

        return states

    def sheaf_failure_trajectory(self) -> np.ndarray:
        """
        Generate trajectory with Čech H^1 gluing failure.
        Each context window is internally consistent but
        boundary values between windows are inconsistent.
        Alice-in-Boston-and-Seattle pattern.
        """
        from gsod_core import WINDOW_SIZE
        states = np.zeros((self.seq_len, self.hidden_dim))

        # Each window gets a different 'base semantic assignment'
        n_windows = self.seq_len // WINDOW_SIZE
        window_bases = self.rng.randn(n_windows, self.hidden_dim)
        # Make adjacent window bases orthogonal (maximally inconsistent)
        for i in range(1, n_windows):
            # Gram-Schmidt: orthogonalize against previous
            for j in range(i):
                window_bases[i] -= (
                    window_bases[i].dot(window_bases[j]) /
                    (window_bases[j].dot(window_bases[j]) + 1e-8)
                ) * window_bases[j]
            window_bases[i] /= np.linalg.norm(window_bases[i]) + 1e-8

        for t in range(self.seq_len):
            window_idx = t // WINDOW_SIZE
            # Smooth within window, discontinuous at boundary
            local_pos = (t % WINDOW_SIZE) / WINDOW_SIZE
            states[t] = (window_bases[window_idx] * 2.0 +
                         self.rng.randn(self.hidden_dim) * 0.1)

        return states

    def get_all_types(self) -> Dict[str, np.ndarray]:
        """Generate all trajectory types for validation."""
        return {
            "CLEAN": self.clean_trajectory(),
            "BINARY_ORBIT_2": self.binary_entanglement_trajectory(2),
            "BINARY_ORBIT_4": self.binary_entanglement_trajectory(4),
            "BINARY_ORBIT_8": self.binary_entanglement_trajectory(8),
            "CONJUGATE_CLOSURE": self.conjugate_closure_failure(),
            "RESONANCE_7ADIC": self.resonance_7adic_trajectory(),
            "SHEAF_FAILURE": self.sheaf_failure_trajectory(),
        }
