import torch
from model import MCDModel
from src.train import train_model
from evaluate import evaluate_model
from data_loader import get_data_loaders

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data
    _, _, test_loader = get_data_loaders()

    # Load trained model weights
    model = MCDModel()
    model.load_state_dict(torch.load('path_to_your_trained_model.pth', map_location=device))
    model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate the model
    test_loss, test_acc, conf_matrix, report = evaluate_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    print(report)

if __name__ == '__main__':
    run()
