# main.py

import torch
from model import MCDModel
from train import train_model
from prune import prune_model
from quantize import quantize_model
from evaluate import evaluate_model
from sensitivity_analysis import layer_wise_sensitivity_analysis
from data_loader import get_data_loaders  # Assuming you have a data loader module

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders()

    # Initialize model
    model = MCDModel()
    model.to(device)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20)

    # Prune the model
    print("Starting pruning...")
    prune_model(model, amount=0.3)

    # Quantize the model
    print("Starting quantization...")
    quantized_model = quantize_model(model)

    # Evaluate the quantized and pruned model
    print("Evaluating quantized pruned model...")
    test_loss, test_acc, conf_matrix, report = evaluate_model(quantized_model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    print(report)

    # Sensitivity analysis
    print("Performing sensitivity analysis...")
    data_iter = iter(test_loader)
    inputs, targets, domain_labels = next(data_iter)
    inputs = inputs.to(device)
    sensitivities = layer_wise_sensitivity_analysis(quantized_model, inputs)
    for layer, sensitivity in sensitivities.items():
        print(f'{layer}: {sensitivity:.6f}')

if __name__ == '__main__':
    main()
