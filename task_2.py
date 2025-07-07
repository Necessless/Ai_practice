import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from basics.datasets import CustomImageDataset
import random
from basics.utils import show_images, show_single_augmentation, show_multiple_augmentations
from basics.extra_augs import RandomErasingCustom, AutoContrast, Posterize


class RandomGaussianBlur:
    """Рандомно блюрит по гауссу часть изображения"""
    def __init__(self, radius=2.0, p=0.5):
        self.radius = radius
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = transforms.ToPILImage()(img)
        return transforms.ToTensor()(img.filter(ImageFilter.GaussianBlur(self.radius)))
    

class RandomPixelDrop:
    """Применяет к тензору маску, удаляя пиксель с вероятностью в 0.5"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        mask = torch.rand_like(img) > 0.5
        return img * mask


class RandomInvert:
    """Рандомно инвертирует изображение"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return transforms.ToTensor()(ImageOps.invert(transforms.ToPILImage()(img)))


def prepare_data():
    root = './data/train'
    dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
    return dataset


def augmentation_pipeline(dataset: CustomImageDataset):
    class_names = dataset.get_class_names()
    images = []
    images.append(dataset[0])  # Гароу
    images.append(dataset[31])  # Генос
    images.append(dataset[61])  # Генос
    images.append(dataset[91])  # Сайтама
    images.append(dataset[121])  # Соник\

    random_gaussian_blur = transforms.Compose([
        transforms.ToTensor(),
        RandomGaussianBlur(1.0)
    ])

    random_pixel_drop = transforms.Compose([
        transforms.ToTensor(),
        RandomPixelDrop(1.0)
    ])

    random_invert = transforms.Compose([
        transforms.ToTensor(),
        RandomInvert(1.0)
    ])

    random_erasing = transforms.Compose([
        transforms.ToTensor(),
        RandomErasingCustom(1.0)
    ])

    random_autocontrast = transforms.Compose([
        transforms.ToTensor(),
        AutoContrast(1.0)
    ])

    random_posterize = transforms.Compose([
        transforms.ToTensor(),
        Posterize(4)
    ])

    augs = [ 
        (random_gaussian_blur, "Random_Gaussian_Blur"),
        (random_pixel_drop, "Random_pixel_drop"),
        (random_invert, "Random_inver"),
        (random_erasing, "Random_erasing"),
        (random_autocontrast, "Random_autocontrast"),
        (random_posterize, "Random_posterize")

    ]

    for image, label in images:
        images = []
        labels = []
        name = class_names[label]
        for augmentation, label_aug in augs:
            new_image = augmentation(image)
            images.append(new_image)
            labels.append(label_aug)
            show_single_augmentation(image, new_image, label_aug, f"results/task_2/{name}_{label_aug}.png")
        show_multiple_augmentations(image, images, labels, f"results/task_2/combined_diff_{name}.png")


if __name__ == "__main__":
    data = prepare_data()
    augmentation_pipeline(data)