# LSTM Chinese Text Generation — Journey to the West

A Chinese text generation project based on LSTM (Long Short-Term Memory). It trains on the original text of *Journey to the West* and learns to generate text in a similar classical style.

## Overview

This project builds a multi-layer LSTM in PyTorch. It uses character-level language modeling to learn the style of *Journey to the West* and generate Chinese text with a classical feel.

## Project Structure

```
LSTM/
├── data/
│   └── xyj.txt   # original text of Journey to the West
├── models/
│   ├── lstm_xiyouji.pth   # trained model weights
│   └── vocab.pkl    # vocabulary mapping
├── lstm-text-generated.ipynb # main notebook
└── README.md   # this file
```

## Features

- **Character-level modeling** — treats each Chinese character as a token, so it can generate any combination of characters
- **Multi-layer LSTM** — 3 LSTM layers for more expressive capacity
- **Dropout** — reduces overfitting and improves generalization
- **Mixed precision** — automatic mixed precision (AMP) speeds up training and lowers memory use
- **Resume training** — continue from a saved checkpoint
- **Temperature sampling** — adjustable temperature controls randomness
- **Multi-GPU** — DataParallel support

## Model Architecture

| Component | Configuration |
|------|------|
| Embedding | Embedding(4198, 256) |
| LSTM | 3 layers, hidden size 512 |
| Dropout | 0.3 |
| Output | Linear(512, 4198) |

**Parameters**: about 9M (34.36 MB)

## Requirements

See the Jupyter notebook for the exact environment and dependencies.

## Usage

### 1. Train the model

Open `lstm-text-generated.ipynb` and run all cells in order.

Key settings (editable in the notebook):

```python
# Data
SEQ_LENGTH = 100      # input sequence length
BATCH_SIZE = 1024     # batch size

# Model
EMBED_DIM = 256       # embedding dimension
HIDDEN_DIM = 512      # LSTM hidden size
NUM_LAYERS = 3        # LSTM layers
DROPOUT = 0.3         # dropout probability

# Training
EPOCHS = 10           # training epochs
LEARNING_RATE = 0.001 # learning rate
CONTINUE_TRAIN = 1    # resume from an existing model

# Generation
GENERATE_LENGTH = 500 # generated text length
TEMPERATURE = 0.1     # sampling temperature (lower = more conservative, higher = more random)
```

### 2. Generate text

After training:

```python
# Load the model
model = load_model_for_inference(MODEL_SAVE_PATH, VOCAB_SAVE_PATH, device)

# Generate
generated_text = generate_text(model, start_text="话说", length=500, device=device, temperature=0.8)
print(generated_text)
```

### 3. Temperature

- **temperature < 0.5** — more conservative and deterministic, closer to the original style
- **temperature ≈ 0.8** — balances creativity and coherence
- **temperature > 1.0** — more random and creative, but grammar errors are more likely

## Training Data

- **Source**: original text of *Journey to the West*
- **Length**: about 735,000 characters
- **Vocabulary**: 4,198 unique characters

## Example Output

Different start texts lead to different styles:

| Start text | Generated style |
|---------|---------|
| 话说 | Narrative opening |
| 悟空 | Sun Wukong-related episodes |
| 唐僧 | Tang Sanzang-related episodes |
| 那妖怪 | Battle scene descriptions |

## Technical Details

### Data Preprocessing

- Remove extra whitespace and special characters
- Build a character-to-index mapping
- Convert text to integer sequences

### Training

- Adam optimizer
- Learning-rate scheduling (ReduceLROnPlateau)
- Gradient clipping (prevents exploding gradients)
- Save the model every epoch

### Generation

- Sample from the softmax distribution
- Temperature controls randomness
- Priming with a start text

## Notes

1. Training on a GPU is recommended; CPU training is slow.
2. Generated text may be ungrammatical in places. Lowering the temperature usually helps.

## License

For learning and research use only.

## Acknowledgements

- Training data: original text of *Journey to the West*
- Built with PyTorch

> Author: Ailan Anjuxi
> Contact:
  Email: anjuxi.ME@outlook.com
  SIP: sip:anjuxi@sip.linphone.org
