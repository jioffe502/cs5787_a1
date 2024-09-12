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

### Training Each Setting

To train the model with each of the regularization settings, follow these steps in the notebook:

1. **Select Configuration**: The configurations are defined in a list within the notebook, specifying whether to use Dropout, Weight Decay, or Batch Normalization.
   
   ```python
   configs = [
       {"name": "No Regularization (GELU)", "dropout": False, "weight_decay": 0, "batch_norm": False},
       {"name": "Dropout (GELU)", "dropout": True, "weight_decay": 0, "batch_norm": False},
       {"name": "Weight Decay (GELU)", "dropout": False, "weight_decay": 1e-4, "batch_norm": False},
       {"name": "Batch Normalization (GELU)", "dropout": False, "weight_decay": 0, "batch_norm": True}
   ]
   ```

2. **Run Training**: For each configuration, the notebook loops through the settings, initializes the model, applies the configuration, and trains the model using the specified hyperparameters. 
   
   ```python
   for config in configs:
       model = LeNet5GELU(use_dropout=config['dropout'], use_batch_norm=config['batch_norm'])
       optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=config['weight_decay'])
       train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, num_epochs=20)
   ```

3. **Save Weights**: After training, the notebook saves the model weights for each configuration to allow for evaluation at a later time.

   ```python
   torch.save(model.state_dict(), f"{config['name']}_weights.pth")
   ```

### Testing with Saved Weights

To test a trained model with saved weights:

1. **Load Model Weights**: Specify the configuration and load the saved weights.

   ```python
   model = LeNet5GELU(use_dropout=True, use_batch_norm=False)  # Example: Dropout setting
   model.load_state_dict(torch.load('Dropout (GELU)_weights.pth'))
   ```

2. **Evaluate on Test Data**: Use the evaluation function provided in the notebook to assess the model's performance on the test dataset.

   ```python
   test_accuracy = evaluate_model(model, test_loader)
   print(f"Test Accuracy: {test_accuracy:.4f}")
   ```

## Hyperparameter Selection

### Training Hyperparameters

- **Batch Size**: Selected from `[32, 64, 128]` based on balancing memory constraints and training stability.
- **Learning Rate**: Chosen from `[0.01, 0.001, 0.0001]` with a focus on stable and convergent training progress.
- **Optimizer**: Adam optimizer was used for its ability to adapt the learning rate and provide faster convergence in practice.

The optimal hyperparameters were selected based on validation accuracy, using a split of 80% training data and 20% validation data.

### Regularization Strategies

- **Dropout Rates**: Experimented with `[0.3, 0.5, 0.7]` to mitigate overfitting, with final rates selected based on validation performance.
- **Weight Decay Values**: Tested values `[1e-3, 1e-4, 1e-5]` to penalize large weights, with selection criteria being the generalization ability on unseen data.
- **Batch Normalization**: Applied directly after convolutional layers to stabilize and speed up training.

## Results

The convergence graphs display the training and testing accuracy for the best configuration across the epochs run in the main script:

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
