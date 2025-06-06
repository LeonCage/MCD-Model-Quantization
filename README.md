# MCD Model Quantization and Pruning

This repository contains code for fine-tuning, pruning, and quantizing a Multi-Column Deep (MCD) neural network model using PyTorch. It includes model evaluation, size comparison, and layer-wise sensitivity analysis with detailed visualizations.

## Project Structure

- `src/` - Source code files including training, evaluation, pruning, quantization, and sensitivity analysis scripts.
- `models/` - Saved model checkpoints before and after pruning/quantization.
- `plots/` - Generated plots such as confusion matrices and sensitivity bar charts.
- `data/` - Dataset folder (if applicable).
- `requirements.txt` - Python dependencies required for the project.

## Features

- Evaluate model performance using loss, accuracy, confusion matrix, and classification report.
- Prune and quantize the MCD model to reduce size and increase efficiency.
- Conduct layer-wise sensitivity analysis to understand parameter importance.
- Visualize results using matplotlib.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/mcd_model_quantization.git
cd mcd_model_quantization
