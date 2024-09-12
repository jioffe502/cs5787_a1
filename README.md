# LeNet5 on FashionMNIST with Regularization Techniques

This project implements the LeNet5 architecture on the FashionMNIST dataset using PyTorch, with the entire implementation contained within a single Jupyter notebook for simplicity. The project explores the effects of various regularization techniques, including Dropout, Weight Decay (L2 regularization), and Batch Normalization, on the model's performance.

## Overview

The implementation compares four configurations of the LeNet5 model with the following settings:

1. **No Regularization (GELU Activation)**: Baseline model without any regularization.
2. **Dropout (GELU Activation)**: Dropout applied in the hidden layers.
3. **Weight Decay (GELU Activation)**: L2 regularization applied during training.
4. **Batch Normalization (GELU Activation)**: Batch Normalization applied after convolutional layers.

Each model was trained with multiple hyperparameters sweeps to find optimal settings.

## Configuration

The sweeps were performed with the following hyperparameter settings:

- **Learning Rates**: `[0.01, 0.001, 0.0001]`
- **Batch Sizes**: `[32, 64, 128]`
- **Number of Epochs**: `20`
- **Dropout Rates**: `[0.3, 0.5, 0.7]`
- **Weight Decay Values**: `[1e-3, 1e-4, 1e-5]`

## Project Structure

- **Notebook**: All code, including data loading, model definition, training, evaluation, and plotting convergence graphs, is contained within a single Jupyter notebook (`cs5787_a1.ipynb`).

## How to Run

### Prerequisites

Ensure you have Python 3.x installed along with the following Python packages:

- PyTorch
- torchvision
- matplotlib
- pandas
- numpy
- PIL

### Dataset Preparation

1. Download the FashionMNIST dataset in gzip format from [here](https://github.com/zalandoresearch/fashion-mnist) or from the course Canvas page.
2. Ensure the dataset files (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `test-images-idx3-ubyte.gz`, `test-labels-idx1-ubyte.gz`) are placed in a folder named `data/` within the project directory.

### Running the Notebook

1. Clone the repository and navigate to the project directory.

   ```bash
   git clone https://github.com/cs5787_a1/cs5787_a1.git
   cd cs5787_a1
   ```

2. Open the Jupyter notebook `cs5787_a1.ipynb` and run the cells sequentially to:

   - Load and preprocess the FashionMNIST dataset.
   - Define and compile the LeNet5 model with GELU activation.
   - Train and evaluate the model with each of the regularization techniques.
   - Plot convergence graphs for each configuration.

3. The notebook also includes cells for saving and loading model weights, allowing you to evaluate saved models on the test dataset directly.

## Results

The convergence graphs display the training and testing accuracy for each configuration across the epochs:

1. **No Regularization (GELU)**
2. **Dropout (GELU)**
3. **Weight Decay (GELU)**
4. **Batch Normalization (GELU)**

These graphs help visualize the impact of each regularization method on model performance. A table summarizing the final accuracies for each setting is included in the notebook.

### Conclusions

- **Dropout** helps in reducing overfitting by randomly deactivating neurons, leading to improved test accuracy.
- **Weight Decay** (L2 regularization) penalizes large weights, which can enhance the model's ability to generalize.
- **Batch Normalization** normalizes the output of a previous activation layer, helping to stabilize and speed up the learning process.

These results underscore the importance of regularization techniques in training deep neural networks.

## References

- LeCun et al., 1998 for the LeNet5 architecture applied to MNIST.
- FashionMNIST dataset: [GitHub Repository](https://github.com/zalandoresearch/fashion-mnist)

For detailed instructions and requirements, please refer to the assignment submission guidelines.
