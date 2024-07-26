# Cortex Core
Implementation of the Transformer neural network architecture model, inspired by the original "Attention is All You Need" paper by Google
## Key Features
- Modularity: The project follows a modular approach, providing separate modules for key components such as activation functions, attention mechanisms, linear layers, and loss functions. This organization enhances code reusability and maintainability.
- Tokenization and Vectorization: The tokenizer and vectorizer modules empower the model to preprocess text data, convert it into numerical representations, and make it suitable for input into the Transformer architecture. This enables efficient handling of natural language data.
- Enhanced Regularization: The regularization module incorporates dropout techniques, a vital method to enhance the model's generalization capabilities, reducing overfitting, and improving its robustness during training.
- Positional Encoding: The preprocessing module includes positional encoding, crucial in capturing sequential information and preserving the order of tokens within the input sequences, thereby aiding the Transformer's understanding of sequential data.
- Attention Mechanisms: The attention module provides advanced attention mechanisms, allowing the model to focus on relevant parts of the input data. This attention mechanism is a key element in the Transformer's ability to handle long-range dependencies efficiently.
- Custom Linear Layers: The linear module offers customizable linear layers, which are fundamental building blocks of the neural network. These layers play a significant role in transforming the input data between different representations, contributing to the model's overall performance.
- C Extensions for Performance: The inclusion of C extensions enhances the computational performance of specific operations, optimizing critical computations and boosting the efficiency of the neural network.

The code is customizable to create custom neural network architectures.

Recursive Neural Network

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
