# Transformers from Scratch

## Overview
This project implements a Transformer model from scratch based on the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). The primary objective is to translate English text into French using a self-built Transformer architecture.

## Features
- Implements Transformer architecture from scratch without external high-level libraries like TensorFlow/Keras or PyTorch's `nn.Transformer`.
- Uses multi-head self-attention and positional encoding.
- Trained on an English-French dataset for machine translation.
- Utilizes teacher forcing for training.
- Implements tokenization and preprocessing steps.

## Prerequisites
To run this project, you need the following dependencies:

```bash
pip install numpy torch torchtext nltk matplotlib tqdm
```

## Dataset
The model is trained on a subset of the **English-French** parallel corpus from datasets such as WMT or a custom dataset.

## Model Architecture
- **Embedding Layer**: Converts input tokens into dense vectors.
- **Positional Encoding**: Adds positional information to embeddings.
- **Multi-Head Self-Attention**: Enables context-aware token representations.
- **Feed-Forward Network**: Fully connected layers for transformation.
- **Layer Normalization & Dropout**: Stabilizes training and prevents overfitting.
- **Decoder with Masked Attention**: Generates output sequences.

## Training
To train the model, run:
```bash
python train.py
```
- The script will preprocess the dataset, train the model, and save checkpoints.
- Training loss and accuracy are logged.

## Evaluation
After training, evaluate the model using:
```bash
python evaluate.py --sentence "Hello, how are you?"
```
Expected output:
```bash
Bonjour, comment allez-vous ?
```

## Results
- Achieved reasonable translation accuracy on test sentences.
- Performance improved with increased training data and hyperparameter tuning.

## Future Improvements
- Train on larger datasets to improve fluency.
- Implement beam search for better translation quality.
- Optimize inference speed with model quantization.

## References
- Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS 2017.

## Author
[Manas Bhunya]

