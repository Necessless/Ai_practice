import sklearn.preprocessing
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pandas as pd
import sklearn


class DatasetFromCSV(Dataset):
    """Класс для создания датасета из csv файла"""
    def __init__(self, csv_file, target, is_logistic: bool = False):
        df = pd.read_csv(csv_file, header=0)  # чтение данных из csv файла
        
        y = df[target].copy()
        x = df.drop(columns=[target])
        cat_cols = x.select_dtypes(include=['object', 'category']).columns
        num_cols = x.select_dtypes(include=['number']).columns
        scaler = sklearn.preprocessing.StandardScaler()  # нормализация данных стандарт скейлером из sklearn
        le = sklearn.preprocessing.LabelEncoder()
        for col in cat_cols:
            x[col] = x[col].fillna('missing')  # заполняем пропущенные категориальные категорией missing
            x[col] = le.fit_transform(x[col])
        x[num_cols] = x[num_cols].fillna(x[num_cols].mean())  # заполняем пропущенные значения средним по столбцу
        x[num_cols] = scaler.fit_transform(x[num_cols])  # нормализуем числовые признаки
        # определяем таргетный столбец
        # присваиваем признаки
        self.X = torch.tensor(x.values, dtype=torch.float32)  # переделываем датафреймы в тензоры
        if is_logistic:
            le_target = sklearn.preprocessing.LabelEncoder()
            self.y = torch.tensor(le_target.fit_transform(y.astype(str)), dtype=torch.long)
        else:
            scaler_target = sklearn.preprocessing.StandardScaler()
            self.y = torch.tensor(scaler_target.fit_transform(y.values.reshape(-1, 1)), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

