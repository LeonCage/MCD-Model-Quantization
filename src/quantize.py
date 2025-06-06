import torch

def quantize_model(model, dtype=torch.qint8):
    """
    Quantize the model weights to reduce model size and improve inference efficiency.
    Args:
        model: PyTorch model to quantize.
        dtype: Target quantization data type (default torch.qint8).
    Returns:
        quantized_model: Quantized PyTorch model.
    """
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    return model
