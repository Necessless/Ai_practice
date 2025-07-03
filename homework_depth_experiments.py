import torch
from utils.experiment_utils import get_mnist_loaders, get_cifar_loaders
from utils.model_utils import FullyConnectedModel, count_parameters, save_model
from utils.experiment_utils import train_model
from utils.visualization_utils import plot_and_save_training_history


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


if __name__ == "__main__":
    base_config_path = 'results/depth_experiments/'
    base_save_path = 'plots/'
    best_loss = float('inf')
    first_model = train_and_plot_model_MNIST(base_config_path + '1_1_model', base_save_path + '1_1_model.png', best_loss, "ADAM", lr=0.01)
    second_model = train_and_plot_model_MNIST(base_config_path + '1_2_model', base_save_path + '1_2_model.png', best_loss, "ADAM", lr=0.01)  # запускаю с АДАМ и lr=0.01 так как при таких входных выдает лучший лосс и точность
    third_model_CIFAR = train_and_plot_model_CIFAR(base_config_path + '1_3_model_CIFAR', base_save_path + '1_3_model_CIFAR_DROPOUT_BATCH.png', best_loss, "ADAM", lr=0.01)
    third_model_MNIST = train_and_plot_model_MNIST(base_config_path + '1_3_model', base_save_path + '1_3_model_DROPOUT_BATCH.png', best_loss, "ADAM", lr=0.01)
    fourth_model = train_and_plot_model_MNIST(base_config_path + '1_4_model', base_save_path + '1_4_model.png', best_loss, "ADAM", lr=0.01)
    fifth_model = train_and_plot_model_MNIST(base_config_path + '1_5_model', base_save_path + '1_5_model.png', best_loss, "ADAM", lr=0.01)