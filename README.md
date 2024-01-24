# DynamO: Dynamic Low Rank Neural Network Optimization

## Overview
DynamO is a software solution aimed at optimizing the efficiency of deep neural networks, particularly focusing on reducing their environmental impact. The software utilizes low-rank neural networks to significantly lower memory and computational requirements. This README provides a quick guide on installing, configuring, and using the DynamO framework.

## Features
- Implementation of dynamic low-rank neural networks.
- Modular and configurable network architecture using TOML files.
- Support for different layer types including `lowRank` and `dense`.
- Customizable optimization strategies for different layer types, including a specialized `DynamicLowRankOptimizer`.
- GUI for easy interaction, training, and number prediction using MNIST dataset.
- Agile development approach ensuring a user-friendly and extendable software.

## Installation
1. **Create a Virtual Environment:** Use `conda` or `venv` to create and activate a new virtual environment.
2. **Install Poetry:** Run `pip install poetry` for dependency management.
3. **Install Dependencies:** Execute `poetry install` in the root directory to set up all necessary packages.

   Alternatively, run `pip install -r requirements.txt` at the root of the project if you encounter issues with Poetry.

## Configuring the Network
Network configuration is done via a TOML file. This file allows you to define the network's structure, including layer types, dimensions, activation functions, and optimizer settings.

Example configuration:

```toml
[settings]
batchSize = 64
numEpochs = 10
architecture = 'FFN'

[[layer]]
type = 'lowRank'
dims = [784, 64]
activation = 'relu'
rank = 20

[[layer]]
type = 'dense'
dims = [64, 10]
activation = 'linear'

[optimizer.default]
type = 'SimpleSGD'
parameters = { lr = 0.005 }

[optimizer.lowRank]
type = 'DynamicLowRankOptimizer'
parameters = { lr = 0.1 }
```

To run the network:

    For GUI usage: Navigate to src/lowrank/main.py and execute the file.
    For custom implementations: Import the desired modules and piece them together as needed.

Sample instantiation code:

```python
model = FeedForward.create_from_config("PATH_TO_YOUR_CONFIG")
trainer = Trainer.create_from_model(model)
trainer.train(your_train_dataloader, your_test_dataloader)
```

Further Documentation

For detailed information on software architecture, modularity, and specific functionalities like GUI and MetaOptimizer, please refer to the included LaTeX documentation.

Acknowledgments

This project is developed as part of coursework in the course INF202 at the Norwegian University of Life Sciences (NMBU).