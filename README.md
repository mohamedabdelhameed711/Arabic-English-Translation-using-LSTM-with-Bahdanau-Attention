# Arabic-English-Translation-using-LSTM-with-Bahdanau-Attention

This project implements a sequence-to-sequence (Seq2Seq) neural machine translation model for translating **Arabic to English** using **LSTM** and **Bahdanau Attention**, built with **PyTorch** and the **Tatoeba dataset**.

---

## 📚 Overview

- **Dataset**: [Tatoeba](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) (Arabic-English subset)
- **Tokenizer**: Pretrained tokenizer from `Helsinki-NLP/opus-mt-ar-en`
- **Architecture**: Encoder-Decoder LSTM with Bahdanau Attention
- **Frameworks**: PyTorch, HuggingFace Datasets, Transformers

---

## 🧠 Model Architecture

- **Encoder**:
  - Embedding layer
  - LSTM layer
- **Decoder**:
  - Embedding layer
  - LSTM with Bahdanau Attention
  - Linear layer for token prediction
- **Attention**:
  - Additive attention mechanism (Bahdanau-style)

---

## 📦 Installation

```
pip install torch transformers datasets`
```

---

## 📁 Dataset Preparation
The dataset is loaded directly using Hugging Face Datasets:

```
from datasets import load_dataset
dataset = load_dataset('Helsinki-NLP/tatoeba_mt', 'ara-eng', split='test[:5%]')
```
Only a 5% subset of the test set is used for faster experimentation.

Tokenization is handled using a pretrained model:

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
```
Both source (Arabic) and target (English) sequences are tokenized and padded manually.

---

## 🏗️ Model Components

➤ Encoder

- Embeds input tokens

- Processes them using an LSTM layer

- Outputs hidden and cell states

➤ Bahdanau Attention

- Computes attention weights based on the decoder hidden state and encoder outputs

- Produces a context vector for the decoder at each timestep

➤ Decoder

- Embeds target input tokens

- Uses the context vector from attention

- Concatenates embedded input and context before passing to LSTM

- Predicts the next token using a linear layer

➤ Seq2Seq

- Combines encoder and decoder

- Uses teacher forcing during training

---

## ⚙️ Training Configuration

Embedding Dim: 256

Hidden Dim: 512

Epochs: 30

Batch Size: 16

Loss Function: CrossEntropyLoss (ignores padding)

Optimizer: Adam

Teacher Forcing Ratio: 0.5

---

## 🏋️‍♂️ Training
To train the model:

```
train()
```
Logs loss per epoch and performs optimization on the translation task.

---

## ✅ Validation
After training, run:

```
validate()
```
Prints the average validation loss over batches.



