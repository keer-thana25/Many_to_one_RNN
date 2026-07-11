# Text Generation using RNN (Many-to-Many)

import os
import pickle

import torch
import torch.nn as nn
import streamlit as st

# ---------------- CONFIGURATION ---------------- #

MODEL = "model/char_rnn_model.pt"
VOCAB = "model/vocab.pkl"

DATA_FILE = "data/text.txt"

SEQ_LEN = 100
BATCH_SIZE = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EMBED_SIZE = 128
EPOCHS = 20
LR = 0.002

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)

# ---------------- MODEL DEFINITION ---------------- #

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)  # one prediction PER TIMESTEP -> many-to-many
        return logits, hidden

# ---------------- DATA HELPERS ---------------- #

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for i, ch in enumerate(chars)}
    return char2idx, idx2char


def get_batch(data, seq_len, batch_size):
    # many-to-many pair: target y is the input shifted by one character
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + seq_len] for s in starts])
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])
    return x.to(DEVICE), y.to(DEVICE)

# ---------------- TRAIN MODEL ---------------- #

def train_model():

    print("Training model...")

    text = load_text(DATA_FILE)
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)

    data = torch.tensor([char2idx[ch] for ch in text], dtype=torch.long)

    model = CharRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = max(1, len(data) // (SEQ_LEN * BATCH_SIZE))

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            x, y = get_batch(data, SEQ_LEN, BATCH_SIZE)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")

        progress_bar.progress(epoch / EPOCHS)
        status_text.text(f"Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # Save model weights
    torch.save(model.state_dict(), MODEL)

    # Save vocab + config needed to rebuild the model later
    with open(VOCAB, "wb") as f:
        pickle.dump(
            {
                "char2idx": char2idx,
                "idx2char": idx2char,
                "vocab_size": vocab_size,
            },
            f,
        )

    print("\nTraining complete. Model and vocab saved.")
    status_text.text("Training complete.")

# ---------------- GENERATE ---------------- #

def generate_text(seed, length, temperature=0.8):

    with open(VOCAB, "rb") as f:
        vocab = pickle.load(f)

    char2idx = vocab["char2idx"]
    idx2char = vocab["idx2char"]
    vocab_size = vocab["vocab_size"]

    model = CharRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
    model.eval()

    # Drop any seed characters the model has never seen
    seed = "".join(ch for ch in seed if ch in char2idx)
    if not seed:
        return None, "Seed text contains no characters the model was trained on."

    chars_generated = list(seed)
    input_seq = torch.tensor([[char2idx[ch] for ch in seed]], dtype=torch.long).to(DEVICE)

    hidden = None
    with torch.no_grad():
        logits, hidden = model(input_seq, hidden)
        last_char_logits = logits[0, -1, :]

        for _ in range(length):
            probs = torch.softmax(last_char_logits / temperature, dim=0)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            chars_generated.append(idx2char[next_idx])

            next_input = torch.tensor([[next_idx]], dtype=torch.long).to(DEVICE)
            logits, hidden = model(next_input, hidden)
            last_char_logits = logits[0, -1, :]

    return "".join(chars_generated), None

# ---------------- TRAIN IF NEEDED ---------------- #

if not os.path.exists(MODEL) or not os.path.exists(VOCAB):
    with st.spinner("No trained model found — training now (this only happens once)..."):
        train_model()

# ---------------- STREAMLIT UI ---------------- #

st.title("Text Generation using RNN")

st.write("Many-to-Many RNN Example")

st.caption(f"Trained on `{DATA_FILE}` | device: `{DEVICE}`")

seed_text = st.text_area("Enter seed text", value="the ", height=100)

col1, col2 = st.columns(2)

with col1:
    gen_length = st.slider("Number of characters to generate", 50, 1000, 300, step=50)

with col2:
    temperature = st.slider(
        "Creativity (temperature)", 0.2, 1.5, 0.8, step=0.1,
        help="Lower = safer/more repetitive, Higher = more random/creative"
    )

if st.button("Generate"):

    if seed_text.strip() == "":
        st.warning("Please enter some seed text.")
    else:
        with st.spinner("Generating..."):
            result, error = generate_text(seed_text, gen_length, temperature)

        if error:
            st.error(error)
        else:
            st.success("Generated Text")
            st.text_area("Output", value=result, height=250)

st.divider()

if st.button("Retrain Model"):
    with st.spinner("Retraining model..."):
        train_model()
    st.success("Model retrained. You can generate text now.")
    # ---------------- STREAMLIT UI ---------------- #

st.set_page_config(
    page_title="Text Generation using RNN",
    page_icon="✍️",
    layout="wide"
)

# ---------- HEADER ----------

st.title("✍️ Text Generation using RNN")
st.markdown("### Many-to-Many Character Level Text Generation")
st.write(
    """
This application generates new text using a **Character-Level Recurrent Neural Network (GRU)**.
The model predicts the next character repeatedly to generate coherent text.
"""
)

# ---------- SIDEBAR ----------

st.sidebar.header("Model Information")

st.sidebar.success("Model Type: GRU RNN")
st.sidebar.write(f"**Dataset:** `{DATA_FILE}`")
st.sidebar.write(f"**Epochs:** {EPOCHS}")
st.sidebar.write(f"**Sequence Length:** {SEQ_LEN}")
st.sidebar.write(f"**Batch Size:** {BATCH_SIZE}")
st.sidebar.write(f"**Hidden Units:** {HIDDEN_SIZE}")
st.sidebar.write(f"**Embedding Size:** {EMBED_SIZE}")
st.sidebar.write(f"**Learning Rate:** {LR}")
st.sidebar.write(f"**Device:** {DEVICE}")

st.sidebar.divider()

st.sidebar.info(
"""
### How it Works

1. Enter a seed text.
2. Select output length.
3. Adjust creativity.
4. Click Generate.
"""
)

# ---------- MAIN LAYOUT ----------

left, right = st.columns([2,1])

with left:

    seed_text = st.text_area(
        "Enter Seed Text",
        value="Alice ",
        height=180
    )

with right:

    gen_length = st.slider(
        "Output Length",
        min_value=50,
        max_value=1000,
        value=300,
        step=50
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.2,
        max_value=1.5,
        value=0.8,
        step=0.1
    )

    st.metric("Temperature", temperature)

# ---------- GENERATE ----------

st.divider()

col1,col2 = st.columns(2)

with col1:

    if st.button("🚀 Generate Text", use_container_width=True):

        if seed_text.strip()=="":

            st.warning("Please enter some seed text.")

        else:

            with st.spinner("Generating text..."):

                result,error = generate_text(
                    seed_text,
                    gen_length,
                    temperature
                )

            if error:

                st.error(error)

            else:

                st.success("Generation Complete")

                st.text_area(
                    "Generated Text",
                    result,
                    height=350
                )

                st.download_button(
                    "📥 Download Output",
                    result,
                    file_name="generated_text.txt",
                    mime="text/plain"
                )

with col2:

    if st.button("🔄 Retrain Model", use_container_width=True):

        with st.spinner("Training Model..."):

            train_model()

        st.success("Training Complete!")

# ---------- MODEL DETAILS ----------

st.divider()

with st.expander("📚 About this Project"):

    st.markdown("""
### Model

- Character-Level GRU
- Many-to-Many RNN
- Embedding Layer
- GRU Layer
- Fully Connected Output Layer

### Training

- Input Sequence → 100 Characters
- Target → Next Character
- Optimizer → Adam
- Loss Function → CrossEntropyLoss

### Text Generation

The model predicts one character at a time.
Each predicted character becomes the next input until the desired length is reached.
""")

# ---------- FOOTER ----------

st.divider()

st.caption("Deep Learning Practical • Character-Level Text Generation using RNN")