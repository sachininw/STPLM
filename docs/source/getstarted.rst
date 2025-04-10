Getting Started
===============

Installation
------------

STPLM installation instructions.

Make sure you have git-lfs installed (https://git-lfs.com).

.. code-block:: bash

    git lfs install
    git clone https://github.com/sachininw/STPLM.git
    cd STPLM
    pip install .



Tips
----

Please note that GPU resources are required for efficient usage of STPLM. Additionally, we strongly recommend tuning hyperparameters for each downstream fine-tuning application as this can significantly boost predictive potential in the downstream task (e.g. max learning rate, learning schedule, number of layers to freeze, etc.).
