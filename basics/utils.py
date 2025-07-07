import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms
import seaborn as sns


def show_images(images, save_path, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]
    
    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow*2, 2))
    if nrow == 1:
        axes = [axes]
    
    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')
    
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def show_single_augmentation(original_img, augmented_img, title="Аугментация", save_path = "results/default.png"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    original_img = torchvision.transforms.ToTensor()(original_img)

    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)
    
    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')
    
    # Аугментированное изображение
    aug_np = aug_resized.numpy().transpose(1, 2, 0)
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def show_multiple_augmentations(original_img, augmented_imgs, titles, save_path = "results/default.png"):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    original_img = torchvision.transforms.ToTensor()(original_img)
    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    
    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')
    
    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_class_sizes_histogramm(info, save_path):
    all_sizes = sum(info.values(), [])
    plt.figure(figsize=(10, 6))
    sns.histplot(all_sizes, bins=30, kde=True, color='skyblue')
    plt.xlabel("Размер изображения (байты)")
    plt.ylabel("Количество изображений")
    plt.title("Гистограмма размеров всех изображений")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_class_sizes_dist(info, save_path):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=[sizes for sizes in info.values()], palette="Set2")
    plt.xticks(ticks=range(len(info)), labels=info.keys(), rotation=15)
    plt.ylabel("Размер изображения (байты)")
    plt.title("Распределение размеров изображений по классам")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_mem_time_metrics(sizes, times, memories, save_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, marker='o')
    plt.title("Время обработки 100 изображений")
    plt.xlabel("Размер изображения")
    plt.ylabel("Время (сек)")

    plt.subplot(1, 2, 2)
    plt.plot(sizes, memories, marker='o', color='red')
    plt.title("Память, используемая при обработке")
    plt.xlabel("Размер изображения")
    plt.ylabel("Память (МБ)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def make_and_save_graph(loss_values, save_path, accuracy_values = None):
    """Функция для отрисовки и сохранения графиков"""
    fig, ax1 = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(loss_values) + 1)
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss_values, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    if accuracy_values:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(epochs, accuracy_values, color=color, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('Loss accuracy by epoch')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
