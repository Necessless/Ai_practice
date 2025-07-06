import torch
from utils.training_utils import get_mnist_loaders, get_cifar_loaders
from models.cnn_models import SimpleCNN, CNNWithResidual, CIFARCNN
from utils.training_utils import train_model, count_parameters
from utils.visualization_utils import plot_training_history, compare_models, visualize_gradient_flow, plot_conf_matrix
from models.fc_models import FCModelSimple, FCModelDeep
from utils.comparison_utils import get_preds_and_labels


def compare_cnn_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    simple_cnn = SimpleCNN(input_channels=1, num_classes=10).to(device)
    residual_cnn = CNNWithResidual(input_channels=1, num_classes=10).to(device)

    print(f"Simple CNN parameters: {count_parameters(simple_cnn)}")
    print(f"Residual CNN parameters: {count_parameters(residual_cnn)}")

    print("Training Simple CNN...")
    simple_history = train_model(simple_cnn, train_loader, test_loader, epochs=5, device=str(device))
    print("Training Residual CNN...")
    residual_history = train_model(residual_cnn, train_loader, test_loader, epochs=5, device=str(device))
    compare_models(simple_history, residual_history, "plots/task_1/two_cnn_comparison.png") 


def compare_all_models_mnist():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    fc_model = FCModelSimple().to(device)
    simple_cnn = SimpleCNN(input_channels=1, num_classes=10).to(device)
    residual_cnn = CNNWithResidual(input_channels=1, num_classes=10).to(device)

    print(f"Simple CNN parameters: {count_parameters(simple_cnn)}")
    print(f"Residual CNN parameters: {count_parameters(residual_cnn)}")
    print(f"FC parameters: {count_parameters(fc_model)}")
    fc_history = train_model(fc_model, train_loader, test_loader, epochs=5, device=str(device))
    print("Training Simple CNN...")
    simple_history = train_model(simple_cnn, train_loader, test_loader, epochs=5, device=str(device))
    print("Training Residual CNN...")
    residual_history = train_model(residual_cnn, train_loader, test_loader, epochs=5, device=str(device))
    compare_models(fc_history, simple_history, "plots/task_1/fc_cnn_comparison_mnist.png", l1="FC_network", l2="CNN_Simple")
    compare_models(simple_history, residual_history, "plots/task_1/two_cnn_comparison_mnist.png", l1="CNN_Simple", l2="Residual_CNN")
    compare_models(fc_history, residual_history, "plots/task_1/fc_residual_cnn_comparison_mnist.png", l1="FC_Network", l2="Residual_CNN")


def build_and_launch_model_mnist(model: str, save_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    match (model):
        case "FC":
            test_model = FCModelSimple().to(device)
            print(f"Simple FC parameters: {count_parameters(test_model)}")
        case "CNN":
            test_model =  SimpleCNN(input_channels=1, num_classes=10).to(device)
            print(f"Simple CNN parameters: {count_parameters(test_model)}")
        case "CNNRES":
            test_model = CNNWithResidual(input_channels=1, num_classes=10).to(device)
            print(f"Residual CNN parameters: {count_parameters(test_model)}")

    print(f"Training {model}...")
    history = train_model(test_model, train_loader, test_loader, epochs=5, device=str(device))
    plot_training_history(history, save_path)


def compare_all_models_cifar():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    fc_model = FCModelDeep().to(device)
    simple_cnn = CIFARCNN(num_classes=10).to(device)
    residual_cnn = CNNWithResidual(input_channels=3, num_classes=10).to(device)

    print(f"Simple CNN parameters: {count_parameters(simple_cnn)}")
    print(f"Residual CNN parameters: {count_parameters(residual_cnn)}")
    print(f"FC parameters: {count_parameters(fc_model)}")

    fc_history = train_model(fc_model, train_loader, test_loader, epochs=5, device=str(device))
    print("Training Simple CNN...")
    simple_history = train_model(simple_cnn, train_loader, test_loader, epochs=5, device=str(device))
    print("Training Residual CNN...")
    residual_history = train_model(residual_cnn, train_loader, test_loader, epochs=5, device=str(device))
    compare_models(fc_history, simple_history, f"plots/task_1/fc_cnn_comparison_cifar.png", l1="FC_network", l2="CNN_Simple")
    compare_models(simple_history, residual_history, f"plots/task_1/two_cnn_comparison_cifar.png", l1="CNN_Simple", l2="Residual_CNN")
    compare_models(fc_history, residual_history, f"plots/task_1/fc_residual_cnn_comparison_cifar.png", l1="FC_Network", l2="Residual_CNN")
    collect_extended_metrics(fc_model, test_loader, device, "plots/task_1/fc_cifar_")
    collect_extended_metrics(simple_cnn, test_loader, device, "plots/task_1/cnn_residual_cifar_")
    collect_extended_metrics(residual_cnn, test_loader, device, "plots/task_1/cnn_regularization_")


def build_and_launch_model_cifar(model: str, save_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders(batch_size=64) 
    test_model = None
    match (model):
        case "FC":
            test_model = FCModelDeep().to(device)
            print(f"Simple FC parameters: {count_parameters(test_model)}")
        case "CNN":
            test_model =  CIFARCNN(num_classes=10).to(device)
            print(f"Simple CNN parameters: {count_parameters(test_model)}")
        case "CNNRES":
            test_model = CNNWithResidual(input_channels=3, num_classes=10).to(device)
            print(f"Residual CNN parameters: {count_parameters(test_model)}")

    print(f"Training {model}...")
    history = train_model(test_model, train_loader, test_loader, epochs=5, device=str(device))
    plot_training_history(history, save_path)


def collect_extended_metrics(model, dataloader, device, save_path):
    y_true, y_pred = get_preds_and_labels(model, dataloader, device)
    plot_conf_matrix(y_pred=y_pred, y_true=y_true, class_names=[str(i) for i in range(10)], save_path=save_path + "conf_matrix.png")
    visualize_gradient_flow(model, save_path + "grad_flow.png")


def test_mnist():
    build_and_launch_model_mnist("FC", "plots/task_1/fc_model_graph.png", "MNIST")
    compare_all_models_mnist()


def test_cifar():
    build_and_launch_model_cifar("FC", "plots/task_1/fc_deep_cifar_model_graph.png")
    compare_all_models_cifar()


if __name__ == "__main__":
    test_mnist()
    test_cifar()