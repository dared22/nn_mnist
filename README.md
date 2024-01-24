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

DynamO offers various modes of operation to suit different user needs. Below are the instructions for each mode:

### Graphical User Interface (GUI) Mode
For an interactive experience with a user-friendly graphical interface, use the GUI mode. This mode is ideal for those who prefer a visual approach to network training and analysis. To launch the GUI, execute the following command in your terminal:
```bash
python main --gui True
```

### Training with Default Configuration
To train a neural network using the default configuration settings provided by the framework, use the base training mode. This mode is useful for quick experiments or initial evaluations. Run the following command to start training with the base configuration:
```bash
python main
```

### Training with Custom Configuration
For more advanced use-cases where you need to customize the network architecture or training parameters, use the custom configuration mode. Replace PATH_TO_YOUR_CONFIG with the actual path to your TOML configuration file. This command allows you to train the network as per your specific requirements:
```bash
python main --config PATH_TO_YOUR_CONFIG
```

For developers or researchers who require a more tailored setup, DynamO provides the flexibility to create custom implementations. You can import the necessary modules from the framework and piece them together according to your project needs. 

Sample instantiation code:

```python
from lowrank.training import FeedForward, Trainer
model = FeedForward.create_from_config("PATH_TO_YOUR_CONFIG")
trainer = Trainer.create_from_model(model)
trainer.train(your_train_dataloader, your_test_dataloader)
```

Further Documentation

For detailed information on software architecture, modularity, and specific functionalities like GUI and MetaOptimizer, please refer to the included LaTeX documentation.

Acknowledgments

This project is developed as part of coursework in the course INF202 at the Norwegian University of Life Sciences (NMBU).