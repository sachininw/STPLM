About
=====

Model Description
-----------------

**STPLM** is a context-aware, attention-based deep learning model pretrained on a large-scale corpus of single-cell spatial transcriptomes to enable cell context-specific predictions in settings with limited data in network biology. During pretraining, STPLM gained a fundamental understanding of network dynamics, encoding network hierarchy in the attention weights of the model in a completely self-supervised manner. With both few-shot learning and fine-tuning with limited task-specific data, STPLM consistently boosted predictive accuracy in a diverse panel of downstream tasks relevant to cellular behavior inference. Overall, STPLM represents a foundational deep learning model pretrained on a large-scale corpus of human single-cell spatial transcriptomes to gain a fundamental understanding of gene network dynamics that can now be democratized to a vast array of downstream tasks to accelerate discovery of key network regulators and candidate therapeutic targets.

In `our manuscript <>`_, we report results for the original 6 layer STPLM model pretrained on Genecorpus-8M.

Application
-----------

The pretrained STPLM model can be used directly for zero-shot learning, by fine-tuning towards the relevant downstream cell classification tasks.

Example applications demonstrated in `our manuscript <>`_ include:

| *Fine-tuning*:
| - Non-cacerous cell type annotation
| - Non-cacerous vs cancerous cell identification
| - Few-shot cell type identification


