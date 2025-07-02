import sklearn.preprocessing
from homework_model_modification import initialize_linreg_model, initialize_logreg_model, ExtendedLinearRegression
import matplotlib.pyplot as plt
from homework_datasets import DatasetFromCSV
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import sklearn
import seaborn as sns
import torch


class HousingDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_and_save_graph(loss_values, save_path, accuracy_values = None):
    """Функция для отрисовки и сохранения графиков"""
    fig, ax1 = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(loss_values) + 1)
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss_values, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    if accuracy_values:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(epochs, accuracy_values, color=color, label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.title('Loss accuracy by epoch')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def test_1():
    lin_loss = initialize_linreg_model(lr=0.001, batch_size=256, optimizer_func="SGD")
    log_loss, log_acc = initialize_logreg_model(lr=0.001, batch_size=256, optimizer_func="SGD")
    make_and_save_graph(lin_loss, 'plots/lin_256_SGD.png')
    make_and_save_graph(log_loss, 'plots/log_256_SGD.png', log_acc)


def test_2():
    lin_loss = initialize_linreg_model(lr=0.01, batch_size=128, optimizer_func="SGD")
    log_loss, log_acc =initialize_logreg_model(lr=0.01, batch_size=128, optimizer_func="SGD")
    make_and_save_graph(lin_loss, 'plots/lin_128_SGD.png')
    make_and_save_graph(log_loss, 'plots/log_128_SGD.png', log_acc)


def test_3():
    lin_loss = initialize_linreg_model(lr=0.001, batch_size=256, optimizer_func="SGD")
    log_loss, log_acc =initialize_logreg_model(lr=0.001, batch_size=256, optimizer_func="SGD")
    make_and_save_graph(lin_loss, 'plots/lin_256_SGD_3.png')
    make_and_save_graph(log_loss, 'plots/log_256_SGD_3.png', log_acc)


def test_4():
    lin_loss = initialize_linreg_model(lr=0.01, batch_size=128, optimizer_func="ADAM")
    log_loss, log_acc =initialize_logreg_model(lr=0.01, batch_size=128, optimizer_func="ADAM")
    make_and_save_graph(lin_loss, 'plots/lin_128_ADAM.png')
    make_and_save_graph(log_loss, 'plots/log_128_ADAM.png', log_acc)


def test_5():
    lin_loss = initialize_linreg_model(lr=0.01, batch_size=128, optimizer_func="RMS")
    log_loss, log_acc = initialize_logreg_model(lr=0.01, batch_size=128, optimizer_func="RMS")
    make_and_save_graph(lin_loss, 'plots/lin_128_RMS.png')
    make_and_save_graph(log_loss, 'plots/log_128_RMS.png', log_acc)


def test_6():
    lin_loss = initialize_linreg_model(lr=0.01, batch_size=128, optimizer_func="ADAM")
    log_loss, log_acc =initialize_logreg_model(lr=0.01, batch_size=128, optimizer_func="ADAM")
    make_and_save_graph(lin_loss, 'plots/lin_128_ADAM-2.png')
    make_and_save_graph(log_loss, 'plots/log_128_ADAM-2.png', log_acc)


def test_7():
    lin_loss = initialize_linreg_model(lr=0.1, batch_size=64, optimizer_func="RMS")
    log_loss, log_acc =initialize_logreg_model(lr=0.1, batch_size=64, optimizer_func="RMS")
    make_and_save_graph(lin_loss, 'plots/lin_64_RMS.png')
    make_and_save_graph(log_loss, 'plots/log_64_RMS.png', log_acc)


def final_test(x: torch.Tensor, y: torch.Tensor):
    model = ExtendedLinearRegression(in_features=12)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    data = HousingDataset(x, y)
    dataloader = DataLoader(data, batch_size=256, shuffle=True)
    # ExtendedLinearRegression.train_model(100, optimizer, criterion, model, dataloader, "L1", True, 20)
    loss_values = ExtendedLinearRegression.train_model(100, optimizer, criterion, model, dataloader, "L2", False, 20)
    torch.save(model.state_dict(), f'models/extended_linreg_final.pth')
    new_model = ExtendedLinearRegression(in_features=12)
    new_model.load_state_dict(torch.load(f'models/extended_linreg_final.pth'))
    new_model.eval()
    make_and_save_graph(loss_values=loss_values, save_path='plots/lin_256_SGD_FINAL.png')


def task_3_2():
    """Изменения только одного датасета для линейной регрессии"""
    df = pd.read_csv('data/housing.csv', header=0)
    # начало feature enginering 
    # заменим longitude и latitude полиномиальными признаками
    poly = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df[['latitude', 'longitude']])
    poly_features = poly.get_feature_names_out(['latitude', 'longitude'])
    df_poly = pd.DataFrame(X_poly, columns=poly_features, index=df.index)
    df = pd.concat([df, df_poly], axis=1)
    df.drop(['longitude', 'latitude'], axis=1, inplace=True)
    df['house_size_score'] = (
        df['population'] +
        df['households'] +
        df['total_bedrooms'] +
        df['total_rooms']
    )
    # Далее разбиваем на категории (например, на 3 группы)
    df['house_size'] = pd.qcut(df['house_size_score'], q=3, labels=['small', 'medium', 'large'])
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(include=['number']).columns
    scaler = sklearn.preprocessing.StandardScaler()  # нормализация данных стандарт скейлером из sklearn
    le = sklearn.preprocessing.LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('missing') # заполняем пропущенные категориальные категорией missing
        df[col] = le.fit_transform(df[col])
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())  # заполняем пропущенные значения средним по столбцу
    df[num_cols] = scaler.fit_transform(df[num_cols])  # нормализуем числовые признаки
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()
    y = df['median_house_value'].copy()
    x = df.drop(columns=['median_house_value'])
    x = torch.tensor(x.values, dtype=torch.float32)
    scaler_target = sklearn.preprocessing.StandardScaler()
    y = torch.tensor(scaler_target.fit_transform(y.values.reshape(-1, 1)), dtype=torch.float32)
    return x, y
    

if __name__ == "__main__":
    """Эксперименты с моделями"""
    # test_1()
    # test_2()
    # test_3()
    # test_4()
    # test_5()
    # test_6()
    # test_7()
    x, y = task_3_2()
    final_test(x, y)