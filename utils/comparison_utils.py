import torch


def get_preds_and_labels(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return y_true, y_pred


def get_activation_data(model, activations):
    """
    Метод для сбора данных после активации первого слоя,
    так как на дальнейших слоях будет трудно интерпретировать
    информацию для визуализации
    """
    def hook(model, input, output):
        activations['conv1'] = output.detach()

    return model.conv1.register_forward_hook(hook)


def log_gradients(model):
    avg_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name:
            if param.grad is not None:
                layers.append(name)
                avg_grads.append(param.grad.abs().mean().item())

    return layers, avg_grads
