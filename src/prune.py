import torch
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """
    Apply global unstructured pruning to the model weights.
    Args:
        model: PyTorch model to prune.
        amount: Fraction of connections to prune (e.g., 0.3 means 30%).
    Returns:
        model: Pruned PyTorch model.
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Remove pruning re-parametrization to make pruning permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    
    return model
