import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from homework_datasets import DatasetFromCSV
from torch.utils.data import DataLoader


class ExtendedLogisticRegression(torch.nn.Module):
    """
    Расширенная логистическая регрессия из torch.nn.
    Для метрик использованы методы из sklearn.
    """
    def __init__(self, in_features, num_parameters):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, num_parameters)

    def forward(self, x):
        return self.linear(x)

    @staticmethod
    def accuracy(y_pred, y_true):
        return (y_pred == y_true).float().mean().item()
    
    # @staticmethod
    # def calculate_precision(TP, FP):
    #     return TP/(TP + FP + 1e-8)  # добавил 1e-8 чтобы избежать деления на 0

    # @staticmethod
    # def calculate_recall(TP, FN):
    #     return TP/(TP + FN + 1e-8)  # добавил 1e-8 чтобы избежать деления на 0

    @staticmethod
    def train_model(epochs: int, optimizer, criterion, model: torch.nn.Linear, dataloader, num_classes: int):
        loss_values = []
        accuracy_values = []
        for epoch in range(1, epochs + 1):
            total_loss = 0
            total_acc = 0
            # TP = torch.zeros(num_classes)
            # FP = torch.zeros(num_classes)
            # FN = torch.zeros(num_classes)
            all_probs = []
            all_true_y = []
            all_preds = []

            for i, (batch_X, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                # Вычисляем вероятности для метрик
                probs = torch.softmax(logits, dim=1)
                y_pred = torch.argmax(logits, dim=1)
                acc = ExtendedLogisticRegression.accuracy(y_pred, batch_y)
                # Здесь делал ручные расчеты метрик, но из за муторности перешёл на автоматические из sklearn
                # for cls in range(num_classes):
                #     y_pred_cls = (y_pred == cls)
                #     y_true = (batch_y == cls)

                #     TP[cls] += (y_pred_cls & y_true).sum().item()
                #     FP[cls] += (y_pred_cls & ~y_true).sum().item()
                #     FN[cls] += (~y_pred_cls & y_true).sum().item()

                total_loss += loss.item()
                total_acc += acc
                all_preds.append(y_pred.detach().cpu())
                all_probs.append(probs.detach().cpu())
                all_true_y.append(batch_y.detach().cpu())


            avg_loss = total_loss / (i + 1)
            avg_acc = total_acc / (i + 1)
            all_probs = torch.cat(all_probs, dim=0).numpy()         
            all_preds = torch.cat(all_preds, dim=0).numpy() 
            all_true_y = torch.cat(all_true_y, dim=0).numpy()
            loss_values.append(avg_loss)
            accuracy_values.append(avg_acc)
            # precision_manual = ExtendedLogisticRegression.calculate_precision(TP, FP)
            # recall_manual = ExtendedLogisticRegression.calculate_recall(TP, FN)
            # f1_score_manual = (2*precision_manual*recall_manual)/(precision_manual + recall_manual + 1e-8)  # Эпсилон также чтобы избежать деления на 0
    
            precision = precision_score(all_true_y, all_preds, average='macro',zero_division=0)
            recall = recall_score(all_true_y, all_preds,  average='macro', zero_division=0)
            f1 = f1_score(all_true_y, all_preds,  average='macro', zero_division=0)
            roc_auc = roc_auc_score(all_true_y, all_probs, multi_class='ovr', average='macro')
   
            if epoch % 10 == 0:
                log_epoch(epoch, avg_loss, acc=avg_acc, precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)
            # if epoch == 100:
            #     visualize_confusion_matrix(all_true_y, all_preds)
        return loss_values, accuracy_values


class ExtendedLinearRegression(torch.nn.Module):
    """Расширенная линейная регрессия из torch.nn"""
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)
    
    @staticmethod
    def train_model(epochs: int, optimizer, criterion, model: torch.nn.Linear, dataloader, regularization_type: str, early_stopping: bool, max_patience: int):
        penalty = 0.5
        best_loss = float('inf')
        loss_values = []
        curr_patience = 0
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for i, (batch_X, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                if regularization_type == "L1":  # применяем l1 регуляризацию
                    sum_parameters = 0.0
                    for parameter in model.parameters():
                        sum_parameters += parameter.abs().sum()
                    loss += penalty * sum_parameters
                elif regularization_type == "L2":  # применяем l2 регуляризацию
                    sum_parameters = 0.0
                    for parameter in model.parameters():
                        sum_parameters += parameter.pow(2).sum()
                    loss += penalty * sum_parameters
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / (i + 1)
            loss_values.append(avg_loss)
            if epoch % 10 == 0:
                log_epoch(epoch, avg_loss)

            if early_stopping:  # реализация early stopping
                if avg_loss < best_loss:  # если лосс за эпоху меньше лучшего лосса, то меняем лучший лосс
                    best_loss = avg_loss
                    curr_patience = 0
                else:
                    curr_patience += 1  # иначе, повышаем счетчик плохих эпох
                if curr_patience == max_patience:  # если "терпение" достигло предела - останавливаем обучение
                    print(f"Early Stopping! Лосс перестал уменьшаться на {epoch} эпохе.\n Лучший лосс за все эпохи: {best_loss}")
                    return loss_values
        return loss_values


def visualize_confusion_matrix(true_y, pred_y):
    conf_matrix = confusion_matrix(true_y, pred_y)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)


def initialize_linreg_model(lr: float, batch_size: int, optimizer_func: str = "SGD"):
    model = ExtendedLinearRegression(in_features=9)
    criterion = torch.nn.MSELoss()
    optimizer = None
    if optimizer_func == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if optimizer_func == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_func == "RMS":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    data = DatasetFromCSV('data/housing.csv', 'median_house_value', False)  
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # ExtendedLinearRegression.train_model(100, optimizer, criterion, model, dataloader, "L1", True, 20)
    loss_values = ExtendedLinearRegression.train_model(100, optimizer, criterion, model, dataloader, "L2", False, 20)
    torch.save(model.state_dict(), f'models/extended_linreg_{batch_size}.pth')
    new_model = ExtendedLinearRegression(in_features=9)
    new_model.load_state_dict(torch.load(f'models/extended_linreg_{batch_size}.pth'))
    new_model.eval()
    return loss_values


def initialize_logreg_model(lr: float, batch_size: int, optimizer_func: str = "SGD"):
    model = ExtendedLogisticRegression(in_features=5, num_parameters=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = None
    if optimizer_func == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if optimizer_func == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_func == "RMS":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    data = DatasetFromCSV('data/Iris.csv', 'Species', True)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    loss_values, accuracy_values = ExtendedLogisticRegression.train_model(100, optimizer, criterion, model, dataloader, 3)
    torch.save(model.state_dict(), f'models/extended_logreg_{batch_size}.pth')
    new_model = ExtendedLogisticRegression(in_features=5, num_parameters=3)
    new_model.load_state_dict(torch.load(f'models/extended_logreg_{batch_size}.pth'))
    new_model.eval()
    return loss_values, accuracy_values


if __name__ == "__main__":
    """Использовал два датасета из sklearn. Модели сохраняются в папку models"""
    initialize_logreg_model(lr=0.1, batch_size=256, optimizer_func="SGD")
    initialize_linreg_model(lr=0.1, batch_size=256, optimizer_func="SGD")