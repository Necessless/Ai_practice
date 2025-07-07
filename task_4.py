from torchvision import transforms
from basics.datasets import CustomImageDataset
from basics.utils import show_images
from basics.extra_augs import AddGaussianNoise, AutoContrast, CutOut, RandomErasingCustom, Posterize


def prepare_data():
    root = './data/train'
    dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
    return dataset


class AugmentationPipeLine:
    def __init__(self):
        self.augs = [transforms.ToTensor()]

    def apply(self, image):
        augmentation = transforms.Compose(self.augs)
        return augmentation(image)

    def add_augmentation(self, aug):
        self.augs.append(aug)

    def remove_augmentation(self, aug):
        self.augs.remove(aug)

    def get_augmentations(self):
        return self.augs


def create_light_aug():
    pipeline = AugmentationPipeLine()
    pipeline.add_augmentation(transforms.RandomHorizontalFlip(0.5))
    pipeline.add_augmentation(transforms.RandomRotation(0.5))
    pipeline.add_augmentation(transforms.RandomVerticalFlip(0.5))
    return pipeline


def create_medium_aug():
    pipeline = AugmentationPipeLine()
    pipeline.add_augmentation(RandomErasingCustom(0.5))
    pipeline.add_augmentation(CutOut(0.5))
    pipeline.add_augmentation(transforms.GaussianBlur(3))
    return pipeline


def create_heavy_aug():
    pipeline = AugmentationPipeLine()
    pipeline.add_augmentation(RandomErasingCustom(0.5))
    pipeline.add_augmentation(CutOut(0.5))
    pipeline.add_augmentation(transforms.GaussianBlur(3))
    pipeline.add_augmentation(AddGaussianNoise(0.5))
    pipeline.add_augmentation(AutoContrast(0.5))
    pipeline.add_augmentation(Posterize(0.5))
    return pipeline


def run_aug(pipeline: AugmentationPipeLine, data, save_path):
    images = []
    for image, label in data:
        images.append(pipeline.apply(image))
    show_images(images, save_path = save_path)


if __name__ == "__main__":
    data = prepare_data()
    light_pipeline = create_light_aug()
    medium_pipeline = create_medium_aug()
    heavy_pipeline = create_heavy_aug()
    run_aug(light_pipeline, data, 'results/task_4/light_augmentation_batch.png')
    run_aug(medium_pipeline, data, 'results/task_4/medium_augmentation_batch.png')
    run_aug(heavy_pipeline, data, 'results/task_4/heavy_augmentation_batch.png')