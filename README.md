# STPLM: Single-Cell Language Training with Spatial Transcriptomics for Cell Identity Understanding.

---
datasets: Xenium 

tags:
- single-cell
- genomics
---
# STPLM
STPLM is a foundational transformer model pretrained on a large-scale corpus of single cell spatial transcriptomes to enable context-aware predictions in settings with limited data in network biology.

- See [our manuscript]() for details of the original model trained on ~8 million transcriptomes in January 2025 and the initial report of our cell classification strategies.

- See [STPLM.readthedocs.io](https://stplm.readthedocs.io) for documentation.

# Model Description
STPLM is a foundational transformer model pretrained on a large-scale corpus of single cell spatial transcriptomes representing human lung and colon tumor tissues. STPLM was originally pretrained in January 2025 on a gene corpus comprised of ~8 million single cell spatial transcriptomes. STPLM specializes in characterizing cells with high mutational burdens (e.g. tumor cells) subjected to high network rewiring. 

Each single cell and cell neighborhood transcriptome is presented to the model as a rank value encoding where genes are ranked by their expression in that cell scaled by their expression across the entire gene corpus (8M). The rank value encoding provides a nonparametric representation of that cell’s transcriptome and takes advantage of the many observations of each gene’s expression across the pretraining corpus to prioritize genes that distinguish cell state. Specifically, this method will deprioritize ubiquitously highly-expressed housekeeping genes by scaling them to a lower rank. Conversely, genes such as transcription factors that may be lowly expressed when they are expressed but highly distinguish cell state will move to a higher rank within the encoding. Furthermore, this rank-based approach may be more robust against technical artifacts that may systematically bias the absolute transcript counts value while the overall relative ranking of genes within each cell remains more stable.

The rank value encoding of each single cell’s transcriptome then proceeds through N layers of transformer encoder units, where N varies dependent on the model size. Pretraining was accomplished using a masked learning objective where 15% of the genes within each transcriptome were masked and the model was trained to predict which gene should be within each masked position in that specific cell state using the context of the remaining unmasked genes. A major strength of this approach is that it is entirely self-supervised and can be accomplished on completely unlabeled data, which allows the inclusion of large amounts of training data without being restricted to samples with accompanying labels.

We detail applications and results in [our manuscript]().

During pretraining, STPLM gained a fundamental understanding of network dynamics, encoding network hierarchy in the model’s attention weights in a completely self-supervised manner. With both few-shot learning and fine-tuning with limited task-specific data, STPLM consistently boosted predictive accuracy in a diverse panel of downstream tasks relevant to cellular behavior inference Overall, STPLM represents a foundational deep learning model pretrained on a large-scale corpus human single cell spatial transcriptomes to gain a fundamental understanding of gene network dynamics that can now be democratized to a vast array of downstream tasks to accelerate discovery of key network regulators and candidate therapeutic targets.

# Application
The pretrained Geneformer model can be used directly for zero-shot learning, by fine-tuning towards the relevant downstream cell identity learning tasks.

Example applications demonstrated in [our manuscript]() include:


*Fine-tuning*:
- Non-cacerous cell type annotation
- Non-cacerous vs cancerous cell identification
- Few-shot cell type identification


# Installation
In addition to the pretrained model, contained herein are functions for tokenizing and collating data specific to single cell spatial transcriptomics, pretraining the model, fine-tuning the model, extracting and plotting cell embeddings, and performing in silico pertrubation with either the pretrained or fine-tuned models. To install (~20s):

```bash

git lfs install
git clone https://github.com/sachininw/STPLM.git
cd STPLM
pip install .
```


Please note that GPU resources are required for efficient usage of STPLM. Additionally, we strongly recommend tuning hyperparameters for each downstream fine-tuning application as this can significantly boost predictive potential in the downstream task (e.g. max learning rate, learning schedule, number of layers to freeze, etc.).
