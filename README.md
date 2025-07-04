# xformer_molecular_embedder

A high-performance molecular embedder built with PyTorch and powered by **Meta's xFormers** library.

This project aims to create high-quality vector representations (embeddings) for chemical molecules from their SMILES strings. These embeddings capture the rich chemical and structural context of a molecule, making them ideal for downstream tasks like:

*   Property prediction (e.g., solubility, toxicity)
*   Virtual screening and similarity search
*   As a starting point for generative models

## Core Idea

The model is a **Transformer Encoder**, trained on a massive dataset of unlabeled molecules using a self-supervised task: **Masked Language Modeling (MLM)**.

1.  **Input**: A SMILES string, like `CCOc1ccccc1`.
2.  **Masking**: We randomly "hide" a part of the molecule: `CCOc1cc[MASK]cc1`.
3.  **Training**: The model's only goal is to predict the correct atom (`c`) that was hidden.
4.  **Result**: By learning to "fix" broken molecules, the model implicitly learns the underlying rules of chemistry, structure, and valency. This learned knowledge is stored in the form of powerful, context-aware **embeddings**.

 
*A simplified diagram of the training process. The model learns to fill in the blank.*

### Why xFormers?

The attention mechanism in Transformers is computationally expensive and memory-hungry. The **xFormers** library provides highly optimized, memory-efficient attention mechanisms and fused kernels. This allows us to:

*   **Train faster**: Significantly reduce training time per epoch.
*   **Use larger models**: Build deeper and more powerful encoders on the same hardware.
*   **Handle longer sequences**: Process larger molecules without running out of memory.

In short, xFormers makes it practical to train state-of-the-art models on consumer-grade hardware.

## Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch
*   NVIDIA GPU with CUDA support is highly recommended.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://gitlab.cspfmba.ru/aseikin/xformer_molecular_embedder.git
    cd xformer_molecular_embedder
    ```

2.  Install dependencies. We recommend using a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should include `torch`, `xformers`, `rdkit-pypi`, `tokenizers`, etc.

### Usage

(This section will be updated as the project progresses)

**1. Training the Embedder:**

```bash
python train.py --data_path path/to/your/smiles_data.csv --output_dir ./models
```

**2. Generating Embeddings:**

A simple script will be provided to load a pre-trained model and generate embeddings for a list of SMILES strings.

```python
from embedder import MolecularEmbedder

# Load a pre-trained model
model = MolecularEmbedder(checkpoint_path="./models/final_model.pt")

# Get an embedding for a molecule
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # Aspirin
embedding_vector = model.get_embedding(smiles)

print(embedding_vector)
```

## Roadmap

-   [x] Initial model architecture and training script.
-   [ ] Pre-training on a large public dataset (e.g., DrugSpaceX, ChEMBL).
-   [ ] Script for easy inference and embedding generation.
-   [ ] Evaluation on downstream benchmark tasks (e.g., MoleculeNet).
-   [ ] Integration with Hugging Face Hub for model sharing.

## Contributing

Contributions are welcome! Please open an issue to discuss any changes or new features.

---