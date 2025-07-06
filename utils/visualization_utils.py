import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.comparison_utils import log_gradients, get_activation_data
from torchvision.utils import make_grid
import torch


def plot_training_history(history, save_path):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def compare_models(fc_history, cnn_history, save_path, l1: str, l2: str):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fc_history['test_accs'], label=l1, marker='o')
    ax1.plot(cnn_history['test_accs'], label=l2, marker='s')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(fc_history['test_losses'], label=l1, marker='o')
    ax2.plot(cnn_history['test_losses'], label=l2, marker='s')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()


def plot_conf_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_gradient_flow(model, save_path):
    layers, grads = log_gradients(model)
    plt.figure(figsize=(10, 5))
    plt.plot(layers, grads, marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Average Gradient value")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_feature_maps(model, save_path):
    """Метод для визуализации feature maps исходя из активаций первого слоя модели"""
    activations = {}
    get_activation_data(model, activations)
    act = activations['conv1'][0]
    fig, axs = plt.subplots(1, 6, figsize=(15, 5))
    for i in range(10):  # первые 10 "фильтров"
        axs[i].imshow(act[i].cpu(), cmap='viridis')
        axs[i].axis('off')
    plt.show()
    plt.savefig(save_path)


def show_first_layer_activations(model, dataloader, save_path=None, device='cpu'):
    images, _ = next(iter(dataloader))
    first_image = images[0:1].to(device)  
    activations = {}

    handle = get_activation_data(model, activations)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        _ = model(first_image)

    handle.remove()

    conv1_activations = activations['conv1'][0].cpu()  

    conv1_activations = (conv1_activations - conv1_activations.min()) / \
                       (conv1_activations.max() - conv1_activations.min() + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    img_to_show = first_image.squeeze().permute(1, 2, 0).cpu().numpy()
    if img_to_show.shape[2] == 1:  # Grayscale
        img_to_show = img_to_show.squeeze()
        ax1.imshow(img_to_show, cmap='gray')
    else:  
        img_to_show = (img_to_show * 0.5) + 0.5 
        ax1.imshow(img_to_show)
    ax1.set_title('Исходное изображение')
    ax1.axis('off')

    n_activations = min(16, conv1_activations.size(0))
    activation_grid = make_grid(
        conv1_activations[:n_activations].unsqueeze(1), 
        nrow=4, 
        normalize=False,  
        scale_each=True
    )
    ax2.imshow(activation_grid.permute(1, 2, 0), cmap='viridis')
    ax2.set_title(f'Активации первого слоя (первые {n_activations} фильтров)')
    ax2.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()