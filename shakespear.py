import torch
import torch.nn as nn
import torch.nn.functional as F

# configuration
BATCH = 32
CTX_LEN = 8
TRAIN_STEPS = 10000
CHECK_EVERY = 300
LR = 1e-2
EVAL_STEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

# load and preprocess text

with open("shakespear.py", encoding="utf-8") as fh:
    raw_text = fh.read()

alphabet = sorted(set(raw_text))
VOCAB = len(alphabet)

char_to_id = {c: i for i, c in enumerate(alphabet)}
id_to_char = {i: c for i, c in enumerate(alphabet)}

def to_ids(s):
    return [char_to_id[c] for c in s]

def to_text(ids):
    return "".join(id_to_char[i] for i in ids)

corpus = torch.tensor(to_ids(raw_text), dtype=torch.long)

split_idx = int(0.9 * len(corpus))
train_tokens = corpus[:split_idx]
val_tokens = corpus[split_idx:]


# batching utility

def sample_batch(mode):
    source = train_tokens if mode == "train" else val_tokens
    starts = torch.randint(0, len(source) - CTX_LEN, (BATCH,))

    x = torch.stack([source[s : s + CTX_LEN] for s in starts])
    y = torch.stack([source[s + 1 : s + CTX_LEN + 1] for s in starts])

    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def compute_loss():
    model.eval()
    stats = {}

    for mode in ("train", "val"):# ------------------
        losses = torch.zeros(EVAL_STEPS)
        for i in range(EVAL_STEPS):
            xb, yb = sample_batch(mode)
            _, l = model(xb, yb)
            losses[i] = l.item()
        stats[mode] = losses.mean()

    model.train()
    return stats


# language model

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.lookup = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.lookup(x)  # (B, T, C)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def sample(self, seed, steps):
        for _ in range(steps):
            logits, _ = self(seed)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            seed = torch.cat([seed, nxt], dim=1)
        return seed


# training

model = BigramLM(VOCAB).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for step in range(TRAIN_STEPS):
    if step % CHECK_EVERY == 0:
        metrics = compute_loss()
        print(
            f"step {step}: "
            f"train loss {metrics['train']:.4f}, "
            f"val loss {metrics['val']:.4f}"
        )

    xb, yb = sample_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# promptting

model.eval()

prompt = input("\nEnter prompt: ")

prompt_ids = torch.tensor(
    [char_to_id[c] for c in prompt],
    dtype=torch.long,
    device=DEVICE
).unsqueeze(0)

with torch.no_grad():
    output = model.sample(prompt_ids, steps=100)

print("\nGenerated text:\n")
print(to_text(output[0].tolist()))


