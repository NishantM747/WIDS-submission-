import torch
import torch.nn as nn
import torch.nn.functional as F

# Model + training parameters
bsz = 16
ctx_len = 128
steps = 5000
check_interval = 500
lr = 3e-4
eval_steps = 200

emb_dim = 256
heads = 4
layers = 4
drop_p = 0.2

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# Load and encode text
with open("shapespear.txt", encoding="utf-8") as f:
    raw_text = f.read()

symbols = sorted(set(raw_text))
vocab_sz = len(symbols)

to_id = {c: i for i, c in enumerate(symbols)}
to_char = {i: c for i, c in enumerate(symbols)}

encode = lambda s: [to_id[c] for c in s]
decode = lambda x: "".join(to_char[i] for i in x)

tokens = torch.tensor(encode(raw_text), dtype=torch.long)
split_idx = int(0.9 * len(tokens))

train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]

# Batch generator
def fetch_batch(split):
    src = train_tokens if split == "train" else val_tokens
    idx = torch.randint(len(src) - ctx_len, (bsz,))
    x = torch.stack([src[i:i+ctx_len] for i in idx])
    y = torch.stack([src[i+1:i+ctx_len+1] for i in idx])
    return x.to(dev), y.to(dev)

@torch.no_grad()
def compute_loss():
    stats = {}
    model.eval()
    for mode in ("train", "val"):
        vals = torch.zeros(eval_steps)
        for i in range(eval_steps):
            xb, yb = fetch_batch(mode)
            _, l = model(xb, yb)
            vals[i] = l.item()
        stats[mode] = vals.mean()
    model.train()
    return stats

# Attention components
class AttentionHead(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.k = nn.Linear(emb_dim, head_dim, bias=False)
        self.q = nn.Linear(emb_dim, head_dim, bias=False)
        self.v = nn.Linear(emb_dim, head_dim, bias=False)

        self.register_buffer(
            "mask", torch.tril(torch.ones(ctx_len, ctx_len))
        )
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.k(x)
        q = self.q(x)

        att = (q @ k.transpose(-2, -1)) * (k.size(-1) ** -0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        v = self.v(x)
        return att @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        dim = emb_dim // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(dim) for _ in range(n_heads)]
        )
        self.out = nn.Linear(emb_dim, emb_dim)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.out(x))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        return self.net(x)

class TransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention(heads)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

#LLM Class
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_sz, emb_dim)
        self.pos_emb = nn.Embedding(ctx_len, emb_dim)

        self.blocks = nn.Sequential(
            *[TransformerLayer() for _ in range(layers)]
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_sz)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=dev))
        x = tok + pos

        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B * T, vocab_sz)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def sample(self, idx, n_tokens):
        for _ in range(n_tokens):
            idx_cond = idx[:, -ctx_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

# Training

model = GPT().to(dev)
optim = torch.optim.AdamW(model.parameters(), lr=lr)

print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
print("Device:", dev)
if dev == "cuda":
    print(torch.cuda.get_device_name(0))

for step in range(steps):
    if step % check_interval == 0 or step == steps - 1:
        res = compute_loss()
        print(
            f"step {step}: "
            f"train {res['train']:.4f}, "
            f"val {res['val']:.4f}"
        )

    xb, yb = fetch_batch("train")
    _, loss = model(xb, yb)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

# Final Output
seed = torch.zeros((1, 1), dtype=torch.long, device=dev)
out = model.sample(seed, 500)[0].tolist()
print(decode(out))
