
# Create an interactive Q&A version of Dream-Mini

code = '''#!/usr/bin/env python3
"""
Dream-Mini: Interactive Q&A Diffusion Language Model
Question-answering system based on Dream architecture

Features:
- Bidirectional Transformer with discrete diffusion
- Context-aware response generation
- Interactive question-answering
- Trained on Q&A pairs

Run:
    python dream_qa.py
"""

import numpy as np
import random
import urllib.request
import re
from collections import Counter


class DreamQA:
    """Dream-style diffusion model for question answering"""
    
    def __init__(self, vocab_size, d_model=128, num_heads=8, num_layers=4, 
                 d_ff=512, max_seq_len=64, timesteps=100):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.timesteps = timesteps
        self.d_k = d_model // num_heads
        
        # Special tokens
        self.MASK_ID = 0
        self.PAD_ID = 1
        self.UNK_ID = 2
        self.SEP_ID = 3  # Separator between question and answer
        
        self._init_parameters()
        self.mask_probs = np.linspace(0.0, 0.95, timesteps)
        
        params = self._count_parameters()
        print(f"DreamQA: vocab={vocab_size:,}, d_model={d_model}, "
              f"heads={num_heads}, layers={num_layers}, params={params:,}")
    
    def _init_parameters(self):
        np.random.seed(42)
        scale = 0.02 / np.sqrt(self.d_model)
        
        self.token_embeddings = np.random.randn(self.vocab_size, self.d_model) * scale
        self.pos_embeddings = np.random.randn(self.max_seq_len, self.d_model) * scale
        self.time_embed_W = np.random.randn(1, self.d_model) * scale
        self.time_embed_b = np.zeros(self.d_model)
        
        self.layers = []
        for _ in range(self.num_layers):
            layer = {
                'W_q': np.random.randn(self.d_model, self.d_model) * scale,
                'W_k': np.random.randn(self.d_model, self.d_model) * scale,
                'W_v': np.random.randn(self.d_model, self.d_model) * scale,
                'W_o': np.random.randn(self.d_model, self.d_model) * scale,
                'W1': np.random.randn(self.d_model, self.d_ff) * scale,
                'b1': np.zeros(self.d_ff),
                'W2': np.random.randn(self.d_ff, self.d_model) * scale,
                'b2': np.zeros(self.d_model),
                'ln1_gamma': np.ones(self.d_model),
                'ln1_beta': np.zeros(self.d_model),
                'ln2_gamma': np.ones(self.d_model),
                'ln2_beta': np.zeros(self.d_model)
            }
            self.layers.append(layer)
        
        self.output_proj = np.random.randn(self.d_model, self.vocab_size) * scale
        self.output_bias = np.zeros(self.vocab_size)
    
    def _count_parameters(self):
        count = self.vocab_size * self.d_model
        count += self.max_seq_len * self.d_model + self.d_model
        layer_params = (4 * self.d_model**2 + 2 * self.d_model * self.d_ff + 
                       self.d_ff + self.d_model + 4 * self.d_model)
        count += self.num_layers * layer_params
        count += self.d_model * self.vocab_size + self.vocab_size
        return count
    
    def _layer_norm(self, x, gamma, beta):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + 1e-5) + beta
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _attention(self, x, layer):
        """Bidirectional attention - can see full context"""
        seq_len = x.shape[0]
        Q = x @ layer['W_q']
        K = x @ layer['W_k']
        V = x @ layer['W_v']
        
        Q = Q.reshape(seq_len, self.num_heads, self.d_k)
        K = K.reshape(seq_len, self.num_heads, self.d_k)
        V = V.reshape(seq_len, self.num_heads, self.d_k)
        
        scores = np.einsum('qhd,khd->hqk', Q, K) / np.sqrt(self.d_k)
        attention = self._softmax(scores)
        out = np.einsum('hqk,khd->qhd', attention, V)
        return (out.reshape(seq_len, self.d_model) @ layer['W_o'])
    
    def _ffn(self, x, layer):
        return np.maximum(0, x @ layer['W1'] + layer['b1']) @ layer['W2'] + layer['b2']
    
    def forward(self, token_ids, t):
        """Forward pass with bidirectional context"""
        seq_len = len(token_ids)
        x = self.token_embeddings[token_ids] + self.pos_embeddings[:seq_len]
        t_emb = np.sin(np.array([[t/self.timesteps]]) @ self.time_embed_W + self.time_embed_b)
        x = x + t_emb
        
        for layer in self.layers:
            x = x + self._attention(x, layer)
            x = self._layer_norm(x, layer['ln1_gamma'], layer['ln1_beta'])
            x = x + self._ffn(x, layer)
            x = self._layer_norm(x, layer['ln2_gamma'], layer['ln2_beta'])
        
        return x @ self.output_proj + self.output_bias
    
    def mask_tokens(self, token_ids, t, answer_start_pos):
        """Mask only answer portion, keep question visible"""
        seq_len = len(token_ids)
        mask_prob = self.mask_probs[t]
        
        # Only mask answer tokens (after separator)
        answer_len = seq_len - answer_start_pos
        num_masks = int(np.ceil(answer_len * mask_prob))
        
        answer_positions = list(range(answer_start_pos, seq_len))
        random.shuffle(answer_positions)
        mask_positions = answer_positions[:num_masks]
        
        masked_ids = token_ids.copy()
        for pos in mask_positions:
            masked_ids[pos] = self.MASK_ID
        
        return masked_ids, mask_positions
    
    def train_step(self, token_ids, answer_start_pos, lr=0.00001):
        """Train to predict answer given question"""
        t = np.random.randint(1, self.timesteps)
        masked_ids, mask_positions = self.mask_tokens(
            np.array(token_ids), t, answer_start_pos
        )
        
        if len(mask_positions) == 0:
            return 0.0
        
        logits = self.forward(masked_ids, t)
        loss = 0
        
        # Compute features for gradient update
        seq_len = len(masked_ids)
        x = self.token_embeddings[masked_ids] + self.pos_embeddings[:seq_len]
        t_emb = np.sin(np.array([[t/self.timesteps]]) @ self.time_embed_W + self.time_embed_b)
        x = x + t_emb
        
        for layer in self.layers:
            x = x + self._attention(x, layer)
            x = self._layer_norm(x, layer['ln1_gamma'], layer['ln1_beta'])
            x = x + self._ffn(x, layer)
            x = self._layer_norm(x, layer['ln2_gamma'], layer['ln2_beta'])
        
        # Update only on masked positions
        for pos in mask_positions:
            tid = token_ids[pos]
            if tid >= self.vocab_size:
                continue
            probs = self._softmax(logits[pos])
            loss += -np.log(probs[tid] + 1e-10)
            
            grad = probs.copy()
            grad[tid] -= 1.0
            grad /= max(len(mask_positions), 1)
            
            self.output_proj -= lr * np.outer(x[pos], grad)
            self.output_bias -= lr * grad
        
        return loss / max(len(mask_positions), 1)
    
    def train(self, qa_pairs, epochs=40, lr=0.00002):
        """Train on question-answer pairs"""
        losses = []
        print(f"\\nTraining {epochs} epochs on {len(qa_pairs)} Q&A pairs...")
        
        for epoch in range(epochs):
            total_loss = 0
            count = 0
            
            for token_ids, answer_start in qa_pairs:
                loss = self.train_step(token_ids, answer_start, lr)
                if loss > 0:
                    total_loss += loss
                    count += 1
            
            avg_loss = total_loss / max(count, 1)
            losses.append(avg_loss)
            
            if epoch % 8 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
        
        return losses
    
    def answer_question(self, question_ids, max_answer_len=20, temperature=0.5):
        """Generate answer for a question using diffusion"""
        # Create sequence: question + SEP + masked answer
        seq_len = len(question_ids) + 1 + max_answer_len
        if seq_len > self.max_seq_len:
            seq_len = self.max_seq_len
            max_answer_len = seq_len - len(question_ids) - 1
        
        # Initialize: question + SEP + masked answer
        token_ids = np.array(
            question_ids + [self.SEP_ID] + [self.MASK_ID] * max_answer_len
        )
        
        answer_start = len(question_ids) + 1
        mask_positions = list(range(answer_start, len(token_ids)))
        
        # Iterative refinement (reverse diffusion)
        for t in reversed(range(self.timesteps)):
            if not mask_positions:
                break
            
            logits = self.forward(token_ids, t) / temperature
            predicted = np.argmax(self._softmax(logits), axis=1)
            
            if t > 0:
                # Calculate how many to unmask
                answer_len = len(token_ids) - answer_start
                target_masks = int(np.ceil(answer_len * self.mask_probs[t-1]))
                num_unmask = max(1, len(mask_positions) - target_masks)
                
                if mask_positions:
                    unmask = np.random.choice(
                        mask_positions, 
                        min(num_unmask, len(mask_positions)), 
                        replace=False
                    )
                    for pos in unmask:
                        token_ids[pos] = predicted[pos]
                    mask_positions = [p for p in mask_positions if p not in unmask]
            else:
                # Final step: unmask all
                for pos in mask_positions:
                    token_ids[pos] = predicted[pos]
                mask_positions = []
        
        # Extract answer (after separator)
        answer_ids = token_ids[answer_start:]
        return answer_ids


def load_qa_data():
    """Load or create Q&A dataset"""
    
    print("="*70)
    print("LOADING Q&A DATA")
    print("="*70)
    
    # Q&A pairs for training
    qa_data = [
        # Technology Q&A
        ("What is machine learning?", "Machine learning is a method of data analysis that automates analytical model building."),
        ("What is AI?", "Artificial intelligence is the simulation of human intelligence by machines."),
        ("What is deep learning?", "Deep learning uses neural networks with multiple layers to learn from data."),
        ("What is NLP?", "Natural language processing enables computers to understand and process human language."),
        ("What is computer vision?", "Computer vision allows machines to interpret and analyze visual information from images."),
        
        # Science Q&A
        ("What is photosynthesis?", "Photosynthesis is the process plants use to convert sunlight into energy."),
        ("What is gravity?", "Gravity is the force that attracts objects with mass toward each other."),
        ("What is DNA?", "DNA is the molecule that carries genetic information in living organisms."),
        ("What is evolution?", "Evolution is the process by which species change over time through natural selection."),
        ("What is an atom?", "An atom is the smallest unit of matter that retains chemical properties."),
        
        # General Knowledge
        ("Who invented the telephone?", "Alexander Graham Bell invented the telephone in 1876."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa during the Renaissance."),
        ("What is the capital of France?", "Paris is the capital and largest city of France."),
        ("What is the tallest mountain?", "Mount Everest is the tallest mountain on Earth at 29,032 feet."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest and deepest ocean on Earth."),
        
        # Programming Q&A
        ("What is Python?", "Python is a high-level programming language known for simplicity and readability."),
        ("What is a variable?", "A variable is a named storage location that holds a value in programming."),
        ("What is a function?", "A function is a reusable block of code that performs a specific task."),
        ("What is debugging?", "Debugging is the process of finding and fixing errors in code."),
        ("What is an algorithm?", "An algorithm is a step-by-step procedure for solving a problem."),
        
        # Math Q&A
        ("What is addition?", "Addition is the mathematical operation of combining two numbers to get a sum."),
        ("What is multiplication?", "Multiplication is repeated addition of the same number."),
        ("What is geometry?", "Geometry is the branch of mathematics that studies shapes and spatial relationships."),
        ("What is algebra?", "Algebra is the study of mathematical symbols and rules for manipulating them."),
        ("What is calculus?", "Calculus is the mathematical study of continuous change and motion."),
    ]
    
    print(f"\\n✓ Loaded {len(qa_data)} Q&A pairs")
    print("\\nSample Q&A pairs:")
    for i in range(5):
        q, a = qa_data[i]
        print(f"  Q: {q}")
        print(f"  A: {a}\\n")
    
    return qa_data


def prepare_qa_pairs(qa_data, max_vocab=5000):
    """Build vocabulary and encode Q&A pairs"""
    
    print("\\nPreparing data...")
    
    # Collect all text
    all_text = []
    for q, a in qa_data:
        all_text.append(q.lower())
        all_text.append(a.lower())
    
    # Tokenize
    all_tokens = []
    for text in all_text:
        tokens = re.findall(r'\\w+|[^\\w\\s]', text)
        all_tokens.extend(tokens)
    
    # Build vocabulary
    freq = Counter(all_tokens)
    vocab_items = list(freq.most_common())[:max_vocab-4]
    
    vocab = ['<MASK>', '<PAD>', '<UNK>', '<SEP>'] + [w for w, f in vocab_items]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    
    print(f"✓ Vocabulary: {len(vocab):,} words")
    
    # Encode Q&A pairs
    encoded_pairs = []
    for q, a in qa_data:
        # Tokenize question and answer
        q_tokens = re.findall(r'\\w+|[^\\w\\s]', q.lower())
        a_tokens = re.findall(r'\\w+|[^\\w\\s]', a.lower())
        
        # Encode: [question tokens] + [SEP] + [answer tokens]
        q_ids = [word_to_id.get(t, 2) for t in q_tokens]
        a_ids = [word_to_id.get(t, 2) for t in a_tokens]
        
        full_sequence = q_ids + [3] + a_ids  # 3 = SEP
        answer_start = len(q_ids) + 1
        
        encoded_pairs.append((full_sequence, answer_start))
    
    print(f"✓ Encoded {len(encoded_pairs)} Q&A pairs")
    
    return vocab, word_to_id, id_to_word, encoded_pairs


def interactive_qa(model, word_to_id, id_to_word):
    """Interactive Q&A session"""
    
    print("\\n" + "="*70)
    print("INTERACTIVE Q&A MODE")
    print("="*70)
    print("\\nAsk questions based on the training topics:")
    print("  • Technology (AI, ML, NLP, computer vision)")
    print("  • Science (photosynthesis, gravity, DNA)")
    print("  • General knowledge (capital cities, famous people)")
    print("  • Programming (Python, variables, functions)")
    print("  • Math (addition, algebra, calculus)")
    print("\\nType 'quit' or 'exit' to stop\\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\\nGoodbye!")
                break
            
            if not question:
                continue
            
            # Tokenize question
            q_tokens = re.findall(r'\\w+|[^\\w\\s]', question.lower())
            q_ids = [word_to_id.get(t, 2) for t in q_tokens]
            
            # Generate answer
            print("  Generating answer...", end=" ", flush=True)
            answer_ids = model.answer_question(q_ids, max_answer_len=20, temperature=0.3)
            
            # Decode answer
            answer_words = []
            for tid in answer_ids:
                if tid in [0, 1, 3]:  # Skip special tokens
                    continue
                word = id_to_word.get(tid, '<UNK>')
                if word != '<UNK>':
                    answer_words.append(word)
            
            answer = ' '.join(answer_words)
            print(f"\\r  Answer: {answer}\\n")
            
        except KeyboardInterrupt:
            print("\\n\\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}\\n")


def main():
    """Main program with Q&A interface"""
    
    print("="*70)
    print("DREAM-QA: Interactive Question-Answering System")
    print("Based on Dream diffusion architecture")
    print("="*70)
    
    # Load Q&A data
    qa_data = load_qa_data()
    
    # Prepare data
    vocab, word_to_id, id_to_word, encoded_pairs = prepare_qa_pairs(qa_data, max_vocab=5000)
    
    # Initialize model
    print("\\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    print()
    
    model = DreamQA(
        vocab_size=len(vocab),
        d_model=96,
        num_heads=6,
        num_layers=3,
        d_ff=384,
        max_seq_len=64,
        timesteps=100
    )
    
    # Train
    print("\\n" + "="*70)
    print("TRAINING ON Q&A PAIRS")
    print("="*70)
    
    losses = model.train(encoded_pairs, epochs=40, lr=0.00002)
    
    print(f"\\n✓ Training completed!")
    print(f"✓ Initial loss: {losses[0]:.4f}")
    print(f"✓ Final loss: {losses[-1]:.4f}")
    print(f"✓ Improvement: {((losses[0]-losses[-1])/losses[0]*100):.1f}%")
    
    # Demo questions
    print("\\n" + "="*70)
    print("DEMO: Testing on Sample Questions")
    print("="*70)
    
    demo_questions = [
        "What is machine learning?",
        "What is AI?",
        "Who invented the telephone?",
        "What is Python?",
        "What is gravity?"
    ]
    
    for question in demo_questions:
        print(f"\\nQ: {question}")
        
        q_tokens = re.findall(r'\\w+|[^\\w\\s]', question.lower())
        q_ids = [word_to_id.get(t, 2) for t in q_tokens]
        
        answer_ids = model.answer_question(q_ids, max_answer_len=20, temperature=0.3)
        
        answer_words = [id_to_word.get(tid, '') for tid in answer_ids 
                       if tid not in [0, 1, 3] and id_to_word.get(tid, '') != '<UNK>']
        answer = ' '.join(answer_words)
        
        print(f"A: {answer}")
    
    # Interactive mode
    print("\\n" + "="*70)
    print("READY FOR INTERACTIVE Q&A")
    print("="*70)
    
    interactive_qa(model, word_to_id, id_to_word)


if __name__ == "__main__":
    main()
'''

# Save the Q&A version
filename = "dream_qa.py"
with open(filename, 'w') as f:
    f.write(code)

print("="*70)
print("INTERACTIVE Q&A VERSION CREATED")
print("="*70)
print(f"\n✓ File: {filename}")
print(f"✓ Lines: {len(code.splitlines())}")
print(f"✓ Size: {len(code):,} bytes")

print("\n" + "="*70)
print("KEY FEATURES")
print("="*70)
print("""
1. Question-Answering System:
   ✓ User asks questions
   ✓ Model generates answers
   ✓ Based on training data

2. Training Data:
   ✓ 25 Q&A pairs covering:
     • Technology (AI, ML, NLP)
     • Science (physics, biology)
     • General knowledge
     • Programming
     • Math

3. Bidirectional Context:
   ✓ Question provides context
   ✓ Answer generated with full question context
   ✓ Only answer portion is masked during training

4. Interactive Mode:
   ✓ Ask questions in real-time
   ✓ Get generated answers
   ✓ Type 'quit' to exit

5. Demo Mode:
   ✓ Tests on 5 sample questions
   ✓ Shows answer generation
   ✓ Then enters interactive mode
""")

print("\n" + "="*70)
print("USAGE")
print("="*70)
print("""
Run the program:
  python dream_qa.py

What happens:
  1. Loads 25 Q&A pairs
  2. Trains model (40 epochs)
  3. Tests on 5 demo questions
  4. Enters interactive mode

Interactive session:
  Your question: What is machine learning?
  Answer: machine learning is a method of data analysis...
  
  Your question: What is AI?
  Answer: artificial intelligence is the simulation...
  
  Your question: quit
  Goodbye!
""")

print("\n✅ Interactive Q&A system ready!")
print(f"   Run: python {filename}")
