import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import numpy as np
from typing import Tuple

class QuantumClassicalNN(nn.Module):
    def __init__(self, n_qubits: int = 4, n_classes: int = 10):
        super().__init__()
        
        # Classical frontend (CNN)
        self.classical_frontend = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, n_qubits)
        )
        
        # Quantum middle layer
        self.n_qubits = n_qubits
        qc = QuantumCircuit(n_qubits)
        feature_map = ZZFeatureMap(n_qubits)
        ansatz = RealAmplitudes(n_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        
        # Create QNN using EstimatorQNN instead of CircuitQNN
        estimator = Estimator()
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator
        )
        self.quantum_layer = TorchConnector(qnn)
        
        # Classical backend (Dense layers)
        self.classical_backend = nn.Sequential(
            nn.Linear(1, 32),  # EstimatorQNN outputs a single value
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.classical_frontend(x)
        x = torch.tanh(x)  # Normalize for quantum input
        x = self.quantum_layer(x)
        x = self.classical_backend(x)
        return x

def get_data_loaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Prepare MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer, 
                criterion: nn.Module, 
                device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def evaluate(model: nn.Module, 
            test_loader: DataLoader, 
            criterion: nn.Module, 
            device: torch.device) -> Tuple[float, float]:
    """Evaluate model performance."""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    
    return test_loss, accuracy

def main():
    # Hyperparameters
    n_qubits = 4
    n_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and move to device
    model = QuantumClassicalNN(n_qubits=n_qubits).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Get data
    train_loader, test_loader = get_data_loaders(batch_size)
    
    # Training loop
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, f'quantum_classical_mnist_epoch_{epoch}.pt')

if __name__ == "__main__":
    main()
