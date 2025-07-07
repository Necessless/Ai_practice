import torch
from torchvision import transforms
from PIL import Image
from basics.datasets import CustomImageDataset
from basics.utils import show_images, show_single_augmentation, show_multiple_augmentations


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
    images.append(dataset[121])  # Соник

    horizontal_flip = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5)
    ])

    random_crop = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(200, padding=20)
    ])

    color_jitter = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.5, hue = 0.3)
    ])

    random_rotation = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15)
    ])

    random_grayscale = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomGrayscale(p=1.0)
    ])
    combined_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(200, padding=20),
        transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.5, hue = 0.3),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=1.0)
    ])

    augs = [ 
        (horizontal_flip, "Random Horizontal_Flip"),
        (random_crop, "Random_Crop"),
        (color_jitter, "Color_Jitter"),
        (random_rotation, "Random_Rotation"),
        (random_grayscale, "Random_Grayscale")
        ]
    
    for image, label in images:
        images = []
        labels = []
        name = class_names[label]
        for augmentation, label_aug in augs:
            new_image = augmentation(image)
            images.append(new_image)
            labels.append(label_aug)
            show_single_augmentation(image, new_image, label_aug, f"results/task_1/{name}_{label_aug}.png")
        new_image = combined_aug(image)
        show_multiple_augmentations(image, images, labels, f"results/task_1/combined_aug_{name}.png")


if __name__ == "__main__":
    data = prepare_data()
    augmentation_pipeline(data)