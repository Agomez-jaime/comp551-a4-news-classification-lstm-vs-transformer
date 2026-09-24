# News Classification — LSTM From Scratch vs. Fine-Tuned Transformer

COMP 551 (Applied Machine Learning), McGill University — Assignment 4.

Compares two approaches to text classification on the AG News dataset
(4-class news topic classification: World, Sports, Business, Sci/Tech).

## Task 1 — LSTM classifier trained from scratch

- A PyTorch `nn.LSTM`-based classifier trained end-to-end on AG News, with no
  pretrained weights.
- Custom `Dataset`/`DataLoader` pipeline with padded, packed sequences
  (`pack_padded_sequence` / `pad_packed_sequence`) for efficient variable-length
  batching.
- 90/10 stratified train/validation split off the official training set, with
  a fixed random seed for reproducibility.

## Task 2 — Fine-tuning a pretrained Transformer

- Tokenization, padding, truncation, and attention-mask construction for a
  pretrained Transformer encoder (via Hugging Face `transformers`/`datasets`).
- Full fine-tuning on the same AG News train/val/test split used for Task 1,
  enabling a direct, apples-to-apples comparison against the from-scratch LSTM.
- Evaluation and plotting of training curves and test performance for both
  models.

## Files

- `code.ipynb` — full notebook: data loading and splitting, Task 1 (LSTM from
  scratch), Task 2 (Transformer fine-tuning), and evaluation/plots for both.
- `writeup.pdf` — assignment report.

## Running it

The notebook pulls AG News via the Hugging Face `datasets` library
(`load_dataset('ag_news')`), so no manual data download is required:
1. `pip install datasets nltk` (done automatically in the first cell).
2. Run all cells in order — Task 1 trains the LSTM, Task 2 fine-tunes the
   Transformer, and later cells compare both.

## Author

Andrea Gómez Jaime
