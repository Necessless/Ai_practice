import matplotlib.pyplot as plt


def plot_and_save_training_history(history, save_path):
    """Визуализирует историю обучения и сохраняет её в пнг файл"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_weight_distribution(model):
    """Визуализация распределения весов модели"""
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weights = param.data.cpu().numpy().flatten()
            plt.hist(weights, bins=100, alpha=0.6)
            plt.title(f"Weight distribution: {name}")
            plt.xlabel("Weight values")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()