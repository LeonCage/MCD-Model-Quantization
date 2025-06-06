import torch
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """
    Apply global unstructured pruning to the model parameters.

    Args:
        model (nn.Module): The neural network model to prune.
        amount (float): The proportion of connections to prune (0 to 1).

    Returns:
        nn.Module: The pruned model.
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model

def remove_pruning(model):
    """
    Remove pruning reparameterization from the model to make pruning permanent.

    Args:
        model (nn.Module): The pruned model.

    Returns:
        nn.Module: The model with pruning removed.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
    return model
