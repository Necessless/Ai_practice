import torch
from utils.experiment_utils import get_mnist_loaders, get_cifar_loaders
from utils.model_utils import FullyConnectedModel, count_parameters, save_model
from utils.experiment_utils import train_model
from utils.visualization_utils import plot_and_save_training_history
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_plot_model_MNIST(config_path: str, save_plot_path: str, best_loss: float, optimizer: str, lr: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    model = FullyConnectedModel(
        config_path=config_path+'.json'
    ).to(device)
    print(f"Model parameters: {count_parameters(model)}")
    history = train_model(model, train_loader, test_loader, epochs=5, lr=lr, device=str(device), optimizer=optimizer) 
    plot_and_save_training_history(history, save_plot_path)
    test_loss = history.get('avg_test_loss', None)
    test_acc = history.get('test_accs', None)
    if test_loss and test_acc and test_loss < best_loss:
        best_loss = test_loss
        save_model(config_path + ".pth", model, optimizer=optimizer, epoch=5, best_test_loss=best_loss, best_test_acc=sum(test_acc)/len(test_acc))


def train_and_plot_model_CIFAR(config_path: str, save_plot_path: str, best_loss: float, optimizer: str, lr: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    model = FullyConnectedModel(
        config_path=config_path+'.json'
    ).to(device)
    print(f"Model parameters: {count_parameters(model)}")
    history = train_model(model, train_loader, test_loader, epochs=5, lr=lr, device=str(device), optimizer=optimizer) 
    plot_and_save_training_history(history, save_plot_path)
    test_loss = history.get('avg_test_loss', None)
    test_acc = history.get('test_accs', None)
    if test_loss and test_acc and test_loss < best_loss:
        best_loss = test_loss
        save_model(config_path + ".pth", model, optimizer=optimizer, epoch=5, best_test_loss=best_loss, best_test_acc=sum(test_acc)/len(test_acc))


def greed_search(sizes: list[int]):
    return product(sizes, repeat=2)


def train_and_plot_model_MNIST_MANUAL(save_plot_path: str, best_loss: float, optimizer: str, lr: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    sizes = [2048, 1024, 512, 256, 128, 64, 32, 16]
    product = greed_search(sizes)
    results = []
    best_comb = None
    for sz1, sz2 in product:
        model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers=[
            {"type": "linear", "size": sz1},
            {"type": "relu"},
            {"type": "linear", "size": sz2},
            {"type": "relu"},
            ]
        ).to(device)

        history = train_model(model, train_loader, test_loader, epochs=1, lr=lr, device=str(device), optimizer=optimizer)  # сделаю 1 эпоху, чтобы не ждать час

        test_loss = history.get('avg_test_loss', None)
        test_acc = history.get('test_accs', None)
        if test_loss and test_acc and test_loss < best_loss:
            best_loss = test_loss
            print(f"Лучшее количество нейронов на первом слое: {sz1}, на втором:{sz2}")
            plot_and_save_training_history(history, save_plot_path)
            best_comb = (sz1, sz2)
            save_model("results/width_experiments/greedSearch_model.pth", model, optimizer=optimizer, epoch=1, best_test_loss=best_loss, best_test_acc=sum(test_acc)/len(test_acc))
        results.append({
            'sz1': sz1,
            'sz2': sz2,
            'acc': float(sum(test_acc)/len(test_acc))  
        })
    display_heatmap(results)


def display_heatmap(model_results):
    df = pd.DataFrame(model_results)
    heatmap_data = df.pivot(index="sz1", columns="sz2", values="acc")  # за метрику для сравнения выбрал accuracy
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Test Accuracy Heatmap")
    plt.xlabel("Размер второго слоя")
    plt.ylabel("Размер первого слоя")
    plt.savefig("plots/greed_search_heatmap_acc.png")
    plt.show()


if __name__ == "__main__":
    base_config_path = 'results/width_experiments/'
    base_save_path = 'plots/'
    best_loss = float('inf')
    first_model = train_and_plot_model_MNIST(base_config_path + '2_1_model', base_save_path + '2_1_model.png', best_loss, "ADAM", lr=0.001)
    second_model = train_and_plot_model_MNIST(base_config_path + '2_2_model', base_save_path + '2_2_model.png', best_loss, "ADAM", lr=0.001)  
    third_model = train_and_plot_model_MNIST(base_config_path + '2_3_model', base_save_path + '2_3_model_DROPOUT_BATCH.png', best_loss, "ADAM", lr=0.001)
    fourth_model = train_and_plot_model_MNIST(base_config_path + '2_4_model', base_save_path + '2_4_model.png', best_loss, "ADAM", lr=0.001)
    greed_search_model = train_and_plot_model_MNIST_MANUAL(base_save_path + '2_greed_search_model.png', best_loss, "ADAM", lr=0.001)