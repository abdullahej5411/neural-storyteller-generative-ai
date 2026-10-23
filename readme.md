# Neural Storyteller: Deep Multimodal Image Captioning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange?logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end multimodal deep learning pipeline that bridges Computer Vision and Natural Language Processing to generate fluent, context-aware descriptions of images. Built with an encoder-decoder Sequence-to-Sequence (Seq2Seq) architecture using a pre-trained **ResNet-50** visual backbone and a multi-layer **LSTM** language model with **Beam Search** decoding.

---

## Architecture Overview

```text
       [ Input Image ] (224x224x3)
              │
              ▼
    ┌───────────────────┐
    │  ResNet-50 (CNN)  │ ◄── Pre-trained on ImageNet (classifier stripped)
    └───────────────────┘
              │
              ▼
       [ 2048-d Vector ]  (Visual Representation)
              │
              ▼
    ┌───────────────────┐
    │ Feature Projection│ ──► Linear layer projecting to LSTM hidden space
    └───────────────────┘
              │
              ▼
    ┌───────────────────┐
    │   2-Layer LSTM    │ ◄── Word Embeddings (512-d) + Visual Features
    │  Caption Decoder  │ ◄── Teacher Forcing during training
    └───────────────────┘
              │
              ▼
    ┌───────────────────┐
    │  Linear + Softmax │ ──► Next-token vocabulary distribution
    └───────────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
[ Greedy Search ]  [ Beam Search (k=5) ]
 (Fast Baseline)   (High-Fidelity Caption)
```

The system employs a two-stage encoder-decoder paradigm:
1. **Visual Encoder (ResNet-50):** Extracts 2048-dimensional dense visual representations from the global average pooling layer of a pre-trained ResNet-50 network. Feature vectors are cached to optimize downstream training efficiency.
2. **Language Decoder (LSTM):** A 2-layer stacked Long Short-Term Memory network (hidden size 1024, dropout 0.5) that conditions caption generation on visual features and preceding word tokens via dense 512-dimensional word embeddings.
3. **Inference Decoders:** Supports deterministic **Greedy Search** for low-latency generation and **Beam Search** ($k=5$) to explore multiple sentence trajectories and prevent repetitive phrasing.

---

## Key Features

- **Transfer Learning Backbone:** Leverages deep spatial hierarchies from ImageNet-pretrained ResNet-50.
- **Robust Tokenization & Vocab Curation:** Frequency-thresholded vocabulary with dedicated `<pad>`, `<start>`, `<end>`, and `<unk>` tokens to handle rare concepts.
- **Stabilized Training:** Teacher forcing with cross-entropy loss (padding index ignored), gradient clipping ($||\mathbf{g}|| \le 5.0$), and learning rate scheduling via `ReduceLROnPlateau`.
- **Comprehensive Evaluation:** Quantitative assessment using BLEU (1–4) and METEOR metrics via NLTK.
- **Interactive Web UI:** Integrated [Gradio](https://gradio.app/) dashboard for live image uploads, side-by-side Greedy vs. Beam Search generation, and confidence score visualization.
- **Hardware Acceleration:** Native support for CUDA GPU and TPU acceleration.

---

## Repository Structure

```text
├── genai-neural-storyteller.ipynb   # Complete pipeline notebook (data, training, demo)
├── parameters.md                    # Detailed model architecture & hyperparameter specs
├── .gitignore                       # Git exclusions for secrets, checkpoints & datasets
└── readme.md                        # Project documentation
```

---

## Technical Specifications & Hyperparameters

| Stage | Hyperparameter | Value | Description |
| :--- | :--- | :--- | :--- |
| **Vision Backbone** | Architecture | ResNet-50 | Pre-trained ImageNet feature extractor |
| | Input Resolution | 224 × 224 | Standard 3-channel RGB image dimensions |
| | Feature Dimension | 2048 | Global Average Pooling output |
| **Language Model** | Embedding Dim | 512 | Word vector representation dimension |
| | Hidden Units | 1024 | Hidden state dimension per LSTM cell |
| | LSTM Depth | 2 | Stacked recurrent layers |
| | Dropout | 0.5 | Regularization between recurrent layers |
| **Training** | Batch Size | 128 | Training samples per iteration |
| | Optimizer | Adam | Learning rate: $3 \times 10^{-4}$ |
| | Scheduler | ReduceLROnPlateau | Factor: 0.5, Patience: 3 epochs |
| | Max Grad Norm | 5.0 | Gradient clipping threshold |
| | Train / Val Split | 90% / 10% | Image-level disjoint split |
| **Inference** | Beam Width ($k$) | 5 | Parallel hypotheses tracked |
| | Max Caption Length | 40 | Maximum generated sequence length |

For an in-depth reference, see [parameters.md](parameters.md).

---

## Dataset

This model is configured for the **Flickr30k** multimodal benchmark:
- **Dataset Size:** ~31,783 images spanning diverse real-world scenes.
- **Ground Truth:** 5 reference human captions per image.
- **Preprocessing:** Resized to $224 \times 224$, normalized with standard ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## Getting Started

### 1. Prerequisites

Clone the repository and install the dependencies:

```bash
git clone https://github.com/abdullahej5411/neural-storyteller-generative-ai.git
cd neural-storyteller-generative-ai
```

Install the required Python packages:

```bash
pip install torch torchvision nltk gradio pandas numpy matplotlib tqdm pillow
```

### 2. Running the Pipeline

Open the interactive notebook to run feature extraction, training, or inference:

```bash
jupyter notebook genai-neural-storyteller.ipynb
```

The notebook will automatically:
1. Verify GPU/TPU availability.
2. Download required NLTK corpora (`punkt`, `wordnet`).
3. Extract and cache ResNet-50 features to `flickr30k_features.pkl`.
4. Train the Seq2Seq decoder with real-time validation tracking.
5. Launch the interactive Gradio interface for real-time testing.

---

## Decoding Strategy Comparison

| Strategy | Mechanism | Characteristics |
| :--- | :--- | :--- |
| **Greedy Search** | Selects $\arg\max P(w_t \mid w_{<t}, I)$ at each step | Extremely fast ($O(T)$), but susceptible to local optima and repetition. |
| **Beam Search ($k=5$)** | Maintains the top-$k$ most probable sequence paths | Produces richer, more coherent, and contextually grounded captions. |

---

## Evaluation Metrics

Model performance is benchmarked using standard NLP captioning criteria:
- **BLEU-1 to BLEU-4:** Measures $n$-gram precision against all 5 ground truth references.
- **METEOR:** Incorporates stem, synonym, and paraphrase matching for alignment with human judgment.

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
2. Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). *Show and Tell: A Neural Image Caption Generator*. CVPR.
3. Young, P., Lai, A., Hodosh, M., & Hockenmaier, J. (2014). *From image descriptions to visual denotations: New benchmarks for semantic monitoring*. TACL.

---

## License

This project is licensed under the [MIT License](LICENSE).
