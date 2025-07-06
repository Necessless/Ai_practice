import torch
from utils.training_utils import get_cifar_loaders
from models.cnn_models import CNNWithResidualCombineCores, CNNWithResidual, CNNBigDepth, CNNMediumDepth, CNNSmallDepth
from utils.training_utils import train_model, count_parameters
from utils.visualization_utils import visualize_feature_maps, compare_models, visualize_gradient_flow, show_first_layer_activations


def compare_all_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    cnn_3x3 = CNNWithResidualCombineCores(3, 10, 3, 3).to(device)
    cnn_5x5 = CNNWithResidualCombineCores(3, 10, 5, 5).to(device)
    cnn_7x7 = CNNWithResidualCombineCores(3, 10, 7, 7).to(device)
    cnn_combine = CNNWithResidualCombineCores(3, 10, 1, 3).to(device)  # модель с двумя сверточными слоями - 1х1 и 3х3

    print(f"CNN 3x3 parameters: {count_parameters(cnn_3x3)}")
    print(f"CNN 5x5 parameters: {count_parameters(cnn_5x5)}")
    print(f"CNN 7x7 parameters: {count_parameters(cnn_7x7)}")
    print(f"CNN combine parameters: {count_parameters(cnn_combine)}")

    print("Training CNN 3x3...")
    cnn_3x3_history = train_model(cnn_3x3, train_loader, test_loader, epochs=5, device=str(device))
    print("Training CNN 5x5...")
    cnn_5x5_history = train_model(cnn_5x5, train_loader, test_loader, epochs=5, device=str(device))
    print("Training CNN 7x7...")
    cnn_7x7_history = train_model(cnn_7x7, train_loader, test_loader, epochs=5, device=str(device))
    print("Training combine CNN 1x1 and 3x3...")
    combine_cnn_history = train_model(cnn_combine, train_loader, test_loader, epochs=5, device=str(device))

    compare_models(cnn_3x3_history, cnn_5x5_history, f"plots/cnn_3x3_5x5_comparison.png", l1="cnn_3x3", l2="cnn_5x5")
    compare_models(cnn_5x5_history, cnn_7x7_history, f"plots/cnn_5x5_7x7_comparison.png", l1="cnn_5x5", l2="cnn_7x7")
    compare_models(cnn_7x7_history, combine_cnn_history, f"plots/cnn_7x7_combine_comparison.png", l1="cnn_7x7", l2="combine_cnn")
    show_first_layer_activations(cnn_3x3, test_loader,"plots/task_2/cnn_3x3_first_layer.png", device)
    show_first_layer_activations(cnn_5x5, test_loader,"plots/task_2/cnn_5x5_first_layer.png", device)
    show_first_layer_activations(cnn_7x7, test_loader, "plots/task_2/cnn_7x7_first_layer.png",device)
    show_first_layer_activations(cnn_combine, test_loader,"plots/task_2/cnn_combine_first_layer.png", device)

    print(f"Receptive fields values: {cnn_3x3.receptive_fields}")
    print(f"Receptive fields values: {cnn_5x5.receptive_fields}")
    print(f"Receptive fields values: {cnn_7x7.receptive_fields}")
    print(f"Receptive fields values: {cnn_combine.receptive_fields}")


def compare_depth():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar_loaders(batch_size=64)
    cnn_small = CNNSmallDepth(3, 10).to(device)
    cnn_medium = CNNMediumDepth(3, 10).to(device)
    cnn_big = CNNBigDepth(3, 10).to(device)
    cnn_residual = CNNWithResidual(3, 10).to(device) 

    print(f"CNN small depth parameters: {count_parameters(cnn_small)}")
    print(f"CNN medium depth parameters: {count_parameters(cnn_medium)}")
    print(f"CNN big depth parameters: {count_parameters(cnn_big)}")
    print(f"CNN residual parameters: {count_parameters(cnn_residual)}")

    print("Training CNN Small depth...")
    cnn_small_history = train_model(cnn_small, train_loader, test_loader, epochs=5, device=str(device))
    print("Training CNN Medium depth...")
    cnn_medium_history = train_model(cnn_medium, train_loader, test_loader, epochs=5, device=str(device))
    print("Training CNN Big Depth...")
    cnn_big_history = train_model(cnn_big, train_loader, test_loader, epochs=1, device=str(device))
    print("Training cnn residual...")
    cnn_residual_history = train_model(cnn_residual, train_loader, test_loader, epochs=5, device=str(device))

    compare_models(cnn_small_history, cnn_medium_history, "plots/cnn_small_medium_comparison.png", l1="cnn_small", l2="cnn_medium")
    compare_models(cnn_medium_history, cnn_big_history, "plots/cnn_medium_big_comparison.png", l1="cnn_medium", l2="cnn_big")
    compare_models(cnn_big_history, cnn_residual_history, "plots/cnn_big_residual_comparison.png", l1="cnn_big", l2="cnn_residual")

    visualize_gradient_flow(cnn_small, "plots/task_2/cnn_small_gradient_flow.png")
    visualize_gradient_flow(cnn_medium, "plots/task_2/cnn_medium_gradient_flow.png")
    visualize_gradient_flow(cnn_big, "plots/task_2/cnn_big_gradient_flow.png")
    visualize_gradient_flow(cnn_residual, "plots/task_2/cnn_residual_gradient_flow.png")

    visualize_feature_maps(cnn_small, "plots/task_2/cnn_small_feature_map.png")
    visualize_feature_maps(cnn_medium, "plots/task_2/cnn_medium_feature_map.png")
    visualize_feature_maps(cnn_big, "plots/task_2/cnn_big_feature_map.png")
    visualize_feature_maps(cnn_residual, "plots/task_2/cnn_residual_feature_map.png")


def test_cifar():
    compare_all_models()  # 2.1
    compare_depth()  # 2.2


if __name__ == "__main__":
    test_cifar()