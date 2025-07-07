from basics.datasets import CustomImageDataset
import os 
from basics.utils import visualize_class_sizes_histogramm, visualize_class_sizes_dist

def prepare_data():
    root = './data/train'
    dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
    return dataset


def count_images_in_class(data: CustomImageDataset):
    classes = data.get_class_names()
    info = {}
    max_size = -9999999
    min_size = 99999999999

    for clas in classes:
        info[clas] = []

    for index, (image, label) in enumerate(data):
        size = os.stat(data.get_raw_image_path(index)).st_size
        if size < min_size:
            min_size = size
        if size > max_size:
            max_size = size
        info[classes[label]].append(size)

    count_classes = [len(x) for x in info.values()]
    mean_classes = [sum(x)/len(x) for x in info.values()]
    mean = sum(mean_classes)/len(mean_classes)

    visualize_class_sizes_histogramm(info, 'results/task_3/sizes_histogramm.png')
    visualize_class_sizes_dist(info, 'results/task_3/class_distrib.png')

    print(f"\nКоличество изображений в классах: {count_classes}\n")
    print(f"\nМинимальный размер изображения во всех классах: {min_size}\n")
    print(f"\nМаксимальный размер изображения во всех классах: {max_size}\n")
    print(f"\nСредний размер изображения во всех классах: {mean}\n")


if __name__ == "__main__":
    data = prepare_data()
    count_images_in_class(data)

