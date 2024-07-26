import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.tokenizer import Tokenizer
from src.activation import ReLU, Softmax
from src.vectorizer import BuildVectors
from src.linear import Linear
from src.loss import Loss

def matmul(input1, input2):
    return np.dot(input1, input2)

class BuildClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BuildClassifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        hidden = self.relu(self.hidden(x))
        output = self.output(hidden)
        return output

    def train_model(self, x, y, learning_rate=0.01, num_epochs=50, batch_size=32):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for i in range(0, len(x), batch_size):
                batch_x = torch.tensor(x[i:i+batch_size], dtype=torch.float32)
                batch_y = torch.tensor(y[i:i+batch_size], dtype=torch.long)
                
                optimizer.zero_grad()
                outputs = self(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss/len(x):.4f}")

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            outputs = self(x)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df['text'], df['sentiment']

def main():
    # Load and preprocess data
    text_data, sentiment_data = load_and_preprocess_data('./data/twitter_training.csv')

    # Encode labels
    label_encoder = Tokenizer()
    sentiment_labels = label_encoder.fit_tokenize(sentiment_data)

    # Vectorize text data
    vectorizer = BuildVectors()
    vectorizer.fit(text_data)
    vectors = vectorizer.transform(text_data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(vectors, sentiment_labels, test_size=0.2, random_state=42)

    # Initialize and train model
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = len(label_encoder.classes)
    model = BuildClassifier(input_size, hidden_size, output_size)
    model.train_model(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes))

    # Example prediction
    def run_prediction(text):
        text_vector = vectorizer.transform([text])
        predicted_index = model.predict(text_vector)[0]
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        return predicted_label

    example_text = "I love this product! It's amazing!"
    print(f"Prediction for '{example_text}': {run_prediction(example_text)}")

if __name__ == "__main__":
    main()


