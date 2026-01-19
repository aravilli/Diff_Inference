"""
Transformer-Based Text Diffusion Model
Inspired by OPT-125m architecture from Facebook/Meta AI
Trains on greeting sentences and generates new greetings from noise

Usage:
    python transformer_text_diffusion.py
"""

import numpy as np
import random
import urllib.request

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


# ============================================================================
# VOCABULARY
# ============================================================================

class TransformerVocab:
    """Vocabulary for tokenizing and detokenizing text"""
    def __init__(self, word_list, max_vocab=200):
        self.words = [w for w, f in word_list[:max_vocab]]
        self.vocab_size = len(self.words)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.words)}

    def encode(self, word):
        return self.word_to_idx.get(word.lower(), 0)

    def decode(self, idx):
        return self.idx_to_word.get(idx, self.words[0])

    def encode_sequence(self, words, max_len=None):
        indices = [self.encode(w) for w in words]
        if max_len and len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices))
        return indices[:max_len] if max_len else indices

    def decode_sequence(self, indices):
        return [self.decode(i) for i in indices]


# ============================================================================
# TRANSFORMER COMPONENTS
# ============================================================================

class MultiHeadAttention:
    """Multi-Head Self-Attention mechanism (like in OPT/GPT)"""
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Weight matrices for Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x):
        """
        Forward pass through multi-head attention
        x: (seq_len, d_model)
        """
        seq_len = x.shape[0]

        # Linear projections
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.num_heads, self.d_k)
        K = K.reshape(seq_len, self.num_heads, self.d_k)
        V = V.reshape(seq_len, self.num_heads, self.d_k)

        # Attention scores
        scores = np.einsum('qhd,khd->hqk', Q, K) / np.sqrt(self.d_k)

        # Causal mask (for autoregressive generation)
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        scores = scores + mask[np.newaxis, :, :]

        # Softmax attention
        attention = self.softmax(scores)

        # Apply attention to values
        out = np.einsum('hqk,khd->qhd', attention, V)

        # Reshape and output projection
        out = out.reshape(seq_len, self.d_model)
        out = out @ self.W_o

        return out

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class FeedForward:
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        out = hidden @ self.W2 + self.b2
        return out


class TransformerBlock:
    """Single Transformer Decoder Block (OPT-style)"""
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layer_norm1_gamma = np.ones(d_model)
        self.layer_norm1_beta = np.zeros(d_model)
        self.layer_norm2_gamma = np.ones(d_model)
        self.layer_norm2_beta = np.zeros(d_model)

    def layer_norm(self, x, gamma, beta):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + 1e-5) + beta

    def forward(self, x):
        # Self-attention with residual connection
        attn_out = self.attention.forward(x)
        x = x + attn_out
        x = self.layer_norm(x, self.layer_norm1_gamma, self.layer_norm1_beta)

        # Feed-forward with residual connection
        ffn_out = self.ffn.forward(x)
        x = x + ffn_out
        x = self.layer_norm(x, self.layer_norm2_gamma, self.layer_norm2_beta)

        return x


class TransformerDenoiser:
    """
    Transformer-based denoiser inspired by OPT-125m
    Predicts noise in diffusion process using self-attention
    """
    def __init__(self, vocab_size, embedding_dim, d_model=64, num_heads=4, 
                 num_layers=3, d_ff=256):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, d_model) * 0.02

        # Positional embeddings
        self.max_seq_len = 12
        self.pos_embeddings = np.random.randn(self.max_seq_len, d_model) * 0.02

        # Time embedding (for diffusion timestep)
        self.time_embed_W = np.random.randn(1, d_model) * 0.02
        self.time_embed_b = np.zeros(d_model)

        # Input projection: embedding_dim -> d_model
        self.input_proj = np.random.randn(embedding_dim, d_model) * 0.02
        self.input_bias = np.zeros(d_model)

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ]

        # Output projection: d_model -> embedding_dim
        self.output_proj = np.random.randn(d_model, embedding_dim) * 0.02
        self.output_bias = np.zeros(embedding_dim)

        print(f"✓ TransformerDenoiser initialized")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - Model dimension: {d_model}")
        print(f"  - Attention heads: {num_heads}")
        print(f"  - Transformer layers: {num_layers}")

    def embed_time(self, t, max_t=100):
        """Embed timestep into feature vector"""
        t_normalized = np.array([[t / max_t]])
        t_emb = np.sin(t_normalized @ self.time_embed_W + self.time_embed_b)
        return t_emb

    def predict_noise(self, xt, t, max_t=100):
        """
        Predict noise using transformer
        xt: (seq_len, embedding_dim)
        Returns: (seq_len, embedding_dim)
        """
        seq_len = xt.shape[0]

        # Time embedding
        time_emb = self.embed_time(t, max_t)

        # Project input to model dimension
        x = xt @ self.input_proj + self.input_bias

        # Add positional embeddings
        pos_emb = self.pos_embeddings[:seq_len]
        x = x + pos_emb

        # Add time conditioning
        x = x + time_emb

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x)

        # Project back to embedding dimension
        predicted_noise = x @ self.output_proj + self.output_bias

        return predicted_noise


# ============================================================================
# DIFFUSION MODEL
# ============================================================================

class DiffusionSchedule:
    """Noise schedule for diffusion process"""
    def __init__(self, timesteps=100):
        self.timesteps = timesteps
        self.betas = np.linspace(0.0001, 0.02, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def get_alpha_bar(self, t):
        return self.alpha_bars[t]


class TransformerSequenceDiffusion:
    """Diffusion model for text sequences"""
    def __init__(self, vocab, schedule, embedding_dim=32, seq_length=8):
        self.vocab = vocab
        self.schedule = schedule
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length

        # Word embeddings
        self.embeddings = np.random.randn(vocab.vocab_size, embedding_dim) * 0.3

    def text_to_embedding(self, text_indices):
        """Convert text indices to embeddings"""
        return self.embeddings[text_indices]

    def embedding_to_text(self, embeddings):
        """Convert embeddings back to text indices (nearest neighbor)"""
        distances = np.sum(
            (self.embeddings[np.newaxis, :, :] - embeddings[:, np.newaxis, :]) ** 2, 
            axis=2
        )
        return np.argmin(distances, axis=1)

    def forward_diffusion(self, x0, t):
        """Add noise to embeddings at timestep t"""
        alpha_bar = self.schedule.get_alpha_bar(t)
        noise = np.random.randn(*x0.shape)
        noisy_x = np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * noise
        return noisy_x, noise

    def denoise_step(self, xt, t, predicted_noise):
        """Single denoising step (reverse diffusion)"""
        alpha = self.schedule.alphas[t]
        alpha_bar = self.schedule.alpha_bars[t]
        beta = self.schedule.betas[t]

        # Predict x0 from noisy xt
        predicted_x0 = (xt - np.sqrt(1 - alpha_bar) * predicted_noise) / np.sqrt(alpha_bar)

        if t > 0:
            noise = np.random.randn(*xt.shape)
            alpha_bar_prev = self.schedule.alpha_bars[t-1]

            # Compute mean
            coef1 = (beta * np.sqrt(alpha_bar_prev)) / (1 - alpha_bar)
            coef2 = (np.sqrt(alpha) * (1 - alpha_bar_prev)) / (1 - alpha_bar)
            mean = coef1 * predicted_x0 + coef2 * xt

            # Compute variance
            variance = ((1 - alpha_bar_prev) / (1 - alpha_bar)) * beta

            return mean + np.sqrt(variance) * noise
        else:
            return predicted_x0


# ============================================================================
# TRAINING
# ============================================================================

def train_transformer_diffusion(diffusion, denoiser, sequences, epochs=100):
    """Train transformer denoiser on text sequences"""
    losses = []

    print(f"\nTraining for {epochs} epochs on {len(sequences)} sequences...")

    for epoch in range(epochs):
        epoch_loss = 0

        for seq_idx in sequences:
            # Convert to embeddings
            x0 = diffusion.text_to_embedding(seq_idx)

            # Random timestep
            t = np.random.randint(0, diffusion.schedule.timesteps)

            # Forward diffusion (add noise)
            xt, true_noise = diffusion.forward_diffusion(x0, t)

            # Predict noise using transformer
            predicted_noise = denoiser.predict_noise(xt, t, diffusion.schedule.timesteps)

            # Compute loss
            loss = np.mean((predicted_noise - true_noise) ** 2)
            epoch_loss += loss

            # Gradient descent update (simplified)
            lr = 0.00003
            error = predicted_noise - true_noise

            # Get transformer features
            seq_len = xt.shape[0]
            time_emb = denoiser.embed_time(t, diffusion.schedule.timesteps)
            x = xt @ denoiser.input_proj + denoiser.input_bias
            pos_emb = denoiser.pos_embeddings[:seq_len]
            x = x + pos_emb + time_emb

            # Forward through transformer
            for block in denoiser.transformer_blocks:
                x = block.forward(x)

            # Update output layer
            grad_w = x.T @ error / (seq_len * error.shape[1])
            grad_b = np.mean(error, axis=0)

            denoiser.output_proj -= lr * grad_w
            denoiser.output_bias -= lr * grad_b

        avg_loss = epoch_loss / len(sequences)
        losses.append(avg_loss)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.6f}")

    return losses


# ============================================================================
# GENERATION
# ============================================================================

def generate_greetings(diffusion, denoiser, num_samples=5):
    """Generate text sequences from random noise"""
    generated = []

    print(f"\nGenerating {num_samples} greeting sequences from noise...\n")

    for sample_idx in range(num_samples):
        print(f"{'='*70}")
        print(f"SAMPLE {sample_idx + 1}")
        print(f"{'='*70}")

        # Start from pure random noise
        seq_len = diffusion.seq_length
        xt = np.random.randn(seq_len, diffusion.embedding_dim)

        print("Starting: Pure random noise")

        # Reverse diffusion process
        key_timesteps = [99, 75, 50, 25, 0]

        for t in reversed(range(diffusion.schedule.timesteps)):
            # Transformer predicts noise
            predicted_noise = denoiser.predict_noise(xt, t, diffusion.schedule.timesteps)

            # Denoise one step
            xt = diffusion.denoise_step(xt, t, predicted_noise)

            # Show progress
            if t in key_timesteps:
                indices = diffusion.embedding_to_text(xt)
                words = diffusion.vocab.decode_sequence(indices)
                clean_words = [w for w in words if w not in ['<PAD>', '<UNK>']]
                print(f"  t={t:3d}: {' '.join(clean_words)}")

        # Final sequence
        final_indices = diffusion.embedding_to_text(xt)
        final_words = diffusion.vocab.decode_sequence(final_indices)
        clean_final = [w for w in final_words if w not in ['<PAD>', '<UNK>']]
        generated.append(clean_final)
        print()

    return generated


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("TRANSFORMER-BASED TEXT DIFFUSION MODEL")
    print("Inspired by OPT-125m architecture")
    print("="*70)

    # Load greetings data
    print("\nLoading greetings data...")
    url = "https://raw.githubusercontent.com/madaan/minimal-text-diffusion/main/data/greetings/greetings.txt"
    with urllib.request.urlopen(url) as response:
        content = response.read().decode('utf-8')

    greetings = [line.strip() for line in content.strip().split('\n') if line.strip()]
    print(f"✓ Loaded {len(greetings)} greetings")

    # Build vocabulary
    print("\nBuilding vocabulary...")
    all_words = []
    for greeting in greetings:
        words = greeting.lower().split()
        all_words.extend(words)

    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    vocab = TransformerVocab(sorted_words, max_vocab=200)
    print(f"✓ Vocabulary size: {vocab.vocab_size}")
    print(f"✓ Top 10 words: {vocab.words[:10]}")

    # Prepare training data
    print("\nPreparing training data...")
    training_greetings = []
    for greeting in greetings:
        words = greeting.lower().split()
        if 4 <= len(words) <= 8:
            training_greetings.append(words)

    random.shuffle(training_greetings)
    training_sample = training_greetings[:40]

    max_len = 8
    encoded_sequences = []
    for seq in training_sample:
        encoded = vocab.encode_sequence(seq, max_len=max_len)
        encoded_sequences.append(encoded)

    print(f"✓ Training sequences: {len(encoded_sequences)}")
    print(f"✓ Sequence length: {max_len}")

    # Print sample training data
    print("\nSample training greetings:")
    for i in range(5):
        print(f"  {i+1}. {' '.join(training_sample[i])}")

    # Initialize models
    print("\nInitializing diffusion model...")
    EMBEDDING_DIM = 32
    schedule = DiffusionSchedule(timesteps=100)
    diffusion = TransformerSequenceDiffusion(
        vocab, schedule, 
        embedding_dim=EMBEDDING_DIM, 
        seq_length=8
    )
    print(f"✓ Embedding dimension: {EMBEDDING_DIM}")
    print(f"✓ Diffusion timesteps: {schedule.timesteps}")

    print("\nInitializing transformer denoiser...")
    denoiser = TransformerDenoiser(
        vocab_size=vocab.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=256
    )

    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    losses = train_transformer_diffusion(
        diffusion, denoiser, encoded_sequences, epochs=100
    )

    print(f"\n✓ Training completed!")
    print(f"✓ Final loss: {losses[-1]:.6f}")
    print(f"✓ Initial loss: {losses[0]:.6f}")
    print(f"✓ Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

    # Generate
    print("\n" + "="*70)
    print("GENERATION")
    print("="*70)
    generated_greetings = generate_greetings(diffusion, denoiser, num_samples=5)

    print("="*70)
    print("FINAL GENERATED GREETINGS")
    print("="*70)
    for i, greeting in enumerate(generated_greetings):
        greeting_text = ' '.join(greeting)
        print(f"{i+1}. {greeting_text}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
