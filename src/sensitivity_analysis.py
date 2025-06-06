import torch
import matplotlib.pyplot as plt

def layer_wise_sensitivity_analysis(model, inputs):
    """
    Perform layer-wise sensitivity analysis by measuring the change in output when
    weights of each layer are perturbed slightly.
    
    Args:
        model: The PyTorch model to analyze.
        inputs: Input tensor for the model.
    
    Returns:
        sensitivities: Dictionary with layer names as keys and sensitivity values as values.
    """
    sensitivities = {}
    model.eval()
    
    # Get the original output
    with torch.no_grad():
        original_output = model(inputs)
    
    # Loop over named parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Save original weights
            original_param = param.data.clone()
            
            # Perturb weights slightly
            epsilon = 1e-3
            param.data.add_(epsilon)
            
            with torch.no_grad():
                perturbed_output = model(inputs)
            
            # Calculate sensitivity as norm of output difference
            sensitivity = (perturbed_output - original_output).abs().mean().item()
            sensitivities[name] = sensitivity
            
            # Restore original weights
            param.data.copy_(original_param)
    
    return sensitivities

def plot_sensitivity(sensitivities):
    """
    Plot bar chart of layer-wise sensitivities.
    
    Args:
        sensitivities: Dictionary of sensitivities.
    """
    layer_names = list(sensitivities.keys())
    sensitivity_values = list(sensitivities.values())

    plt.figure(figsize=(12,8))
    plt.bar(layer_names, sensitivity_values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Layers and Parameters', fontsize=14)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.title('Layer-wise Sensitivity Analysis', fontsize=16)
    plt.tight_layout()
    plt.show()
