from torchvision import transforms
from basics.datasets import CustomImageDataset
from basics.extra_augs import CutOut
from task_4 import AugmentationPipeLine
import time 
import psutil 
import os
from basics.utils import visualize_mem_time_metrics


def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # MB


def metrics_collecter(func):

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_mem = get_memory_usage()
        res = func(*args, **kwargs)
        end_time = time.perf_counter()
        end_mem = get_memory_usage()
        return (end_time-start_time), (end_mem-start_mem), res
    return wrapper


def prepare_data(target_size):
    root = './data/train'
    dataset = CustomImageDataset(root, transform=None, target_size=target_size)
    return dataset 

@metrics_collecter
def test_64x64(augmentation: AugmentationPipeLine):
    data_64x64 = prepare_data((64, 64))
    for i in range(100):  # ограничиваем до 100 картинок
        image, label = data_64x64[i]
        new_image = augmentation.apply(image)


@metrics_collecter
def test_128x128(augmentation: AugmentationPipeLine):
    data_128x128 = prepare_data((128, 128))
    for i in range(100):  # ограничиваем до 100 картинок
        image, label = data_128x128[i]
        new_image = augmentation.apply(image)


@metrics_collecter
def test_224x224(augmentation: AugmentationPipeLine):
    data_224x224 = prepare_data((224, 224))
    for i in range(100):  # ограничиваем до 100 картинок
        image, label = data_224x224[i]
        new_image = augmentation.apply(image)


@metrics_collecter
def test_512x512(augmentation: AugmentationPipeLine):
    data_512x512 = prepare_data((512, 512))
    for i in range(100):  # ограничиваем до 100 картинок
        image, label = data_512x512[i]
        new_image = augmentation.apply(image)


if __name__ == "__main__":
    augmentation = AugmentationPipeLine()
    augmentation.add_augmentation(transforms.RandomHorizontalFlip(1.0))
    augmentation.add_augmentation(transforms.RandomGrayscale(1.0))
    augmentation.add_augmentation(transforms.ColorJitter(brightness=0.5, hue=0.3, saturation=0.4, contrast=0.5))
    augmentation.add_augmentation(CutOut(1.0))
    
    time_64x64, mem_64x64, res = test_64x64(augmentation)
    time_128x128, mem_128x128, res = test_128x128(augmentation)
    time_224x224, mem_224x224, res = test_224x224(augmentation)
    time_512x512, mem_512x512, res = test_512x512(augmentation)
    times = [time_64x64, time_128x128, time_224x224, time_512x512]
    memories = [mem_64x64, mem_128x128, mem_224x224, mem_512x512]
    sizes = ['64x64', '128x128', '224x224', '512x512']
    visualize_mem_time_metrics(sizes, times, memories, 'results/task_5/comparison_graphs.png')
