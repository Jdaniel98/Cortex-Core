# Cortex Core

## Overview
Cortex Core is an implementation of advanced neural network architectures, inspired by the groundbreaking "Attention is All You Need" paper by Google. This project aims to provide a flexible and efficient framework for building and experimenting with state-of-the-art natural language processing models.

## Key Features

### Modular Architecture
- **Separate Modules**: The project is organized into distinct modules for activation functions, attention mechanisms, linear layers, loss functions, and more. This modular approach enhances code reusability and maintainability.

### Text Processing
- **Tokenization and Vectorization**: Custom tokenizer and vectorizer modules enable efficient preprocessing of text data, converting it into numerical representations suitable for neural network input.

### Advanced Neural Network Components
- **Enhanced Regularization**: Incorporates dropout techniques to improve model generalization and reduce overfitting.
- **Positional Encoding**: Crucial for capturing sequential information in input data.
- **Attention Mechanisms**: Enables the model to focus on relevant parts of the input, handling long-range dependencies efficiently.
- **Custom Linear Layers**: Fundamental building blocks for transforming data between different representations.

### Performance Optimization
- **C Extensions**: Critical operations are optimized using C extensions, boosting computational efficiency.

### Customizable Architecture
- The modular design allows for easy customization and experimentation with different neural network architectures.

## Recent Updates
- Implemented a Recurrent Neural Network (RNN) module with improved compatibility with PyTorch.
- Enhanced the RNN implementation to handle hidden states more effectively.

## Example: RNN Implementation

```python
import torch
import torch.nn as nn

from src.preprocessing import Embedding
from src.linear import Linear
from src.activation import ReLU

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = Linear(hidden_dim, output_dim)
        self.activation = ReLU()

    def forward(self, text, hidden=None):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output[:, -1, :])  # Use the last output for classification
        return self.activation(output), hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn.hidden_size)
```

