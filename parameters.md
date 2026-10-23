# Neural Storyteller — Parameters & Hyperparameters

## Feature Extraction (Part 1)

| Parameter | Value | Description |
|-----------|-------|-------------|
| ResNet model | ResNet50 | Pre-trained on ImageNet |
| Feature dimension | 2048 | Output of ResNet50 (classifier removed) |
| Image size | 224 x 224 | Required input for ResNet |
| Normalization mean | (0.485, 0.456, 0.406) | ImageNet standard |
| Normalization std | (0.229, 0.224, 0.225) | ImageNet standard |
| Extraction batch size | 64 | Images processed per batch |

## Vocabulary (Part 2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| MIN_WORD_FREQ | 3 | Words appearing fewer times become `<unk>` |
| Special tokens | `<pad>`, `<start>`, `<end>`, `<unk>` | 4 reserved tokens |

## Model Architecture (Part 3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| EMBED_SIZE | 512 | Word embedding dimension |
| HIDDEN_SIZE | 1024 | LSTM hidden state dimension |
| NUM_LAYERS | 2 | Number of LSTM layers |
| DROPOUT | 0.5 | Dropout probability |

## Training (Part 4)

| Parameter | Value | Description |
|-----------|-------|-------------|
| NUM_EPOCHS | 30 | Total training epochs |
| LEARNING_RATE | 3e-4 | Adam optimizer learning rate |
| BATCH_SIZE | 128 | Training batch size |
| CLIP_GRAD | 5.0 | Max gradient norm (clipping) |
| Train/Val split | 90% / 10% | Split by image (not caption) |
| Loss function | CrossEntropyLoss | Ignores `<pad>` tokens |
| Optimizer | Adam | Standard adaptive optimizer |
| Scheduler | ReduceLROnPlateau | factor=0.5, patience=3 |

## Inference (Part 5)

| Parameter | Value | Description |
|-----------|-------|-------------|
| MAX_CAP_LEN | 40 | Max words generated per caption |
| Beam width | 5 | Number of beams in beam search |
| SEED | 42 | Random seed for reproducibility |
