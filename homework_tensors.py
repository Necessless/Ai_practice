import torch


class Task_1_1:
    """Задание 1.1 - Создание тензоров"""
    @staticmethod
    def run():
        print("\nЗадание 1.1\n")
        tensor_1 = torch.rand(3, 4)  # Тензор размером 3x4, заполненный случайными числами от 0 до 1
        tensor_2 = torch.zeros(2, 3, 4)  # Тензор размером 2x3x4, заполненный нулями
        tensor_3 = torch.ones(5, 5)  # Тензор размером 5x5, заполненный единицами
        tensor_4 = torch.reshape(torch.arange(0, 16), (4, 4))  # Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)

        print(
            f"\nТензор 1:\n {tensor_1}\n" +
            f"Тензор 2:\n {tensor_2}\n" +
            f"Тензор 3:\n {tensor_3}\n" +
            f"Тензор 4:\n {tensor_4}\n"
            )


class Task_1_2:
    """Задание 1.2 - Операции с тензорами"""
    def __init__(self):
        """
        Создаем начальные тензоры со случайными элементами [0,100],
        А также размером 3х4 и 4х3 соответственно
        """
        self.A = torch.randint(0, 100, size=(3, 4))
        self.B = torch.randint(0, 100, size=(4, 3))
        print("\nЗадание 1.2\n")

    def __call__(self):
        transposed_A = self.A.T
        matrix_prod = self.A @ self.B
        element_prod = self.A * self.B.T
        sum_A = self.A.sum()
        print(
            f"\nТранспонированная матрица А:\n {transposed_A}\n" +
            f"Матричное произведение А и В:\n {matrix_prod}\n" +
            f"Поэлементное умножение А и транспонированной В:\n {element_prod}\n" +
            f"Сумма элементов в А:\n {sum_A}\n"
            )


class Task_1_3:
    """Задание 1.3 - Индексация и срезы"""
    def __init__(self):
        """
        Создаем начальный тензор размером 5х5х5 со значениями [0,1)
        """
        self.A = torch.rand(8, 8, 8)
        print("\nЗадание 1.3\n")

    def __call__(self):
        """
        Не совсем понял как нужно было выводить первую строку 
        и последний столбец, во всём тензоре или в каждом его измерении.
        Поэтому сделал и так и так.
        """
        tensor_center = self.A[2, 1:3, 1:3] 
        print(
            f"\nТензор А: {self.A}\n" +
            f"Первая строка в каждом измерении в А:\n {self.A[:, 0, :]}\n" +
            f"Последний столбец в каждом измерении в А:\n {self.A[:, :, -1]}\n"
            f"Первая строка в целом в А:\n {self.A[0, 0, :]}\n" +
            f"Последний столбец в целом в А:\n {self.A[-1, :, -1]}\n" +
            f"Подматрица 2х2 из центра:\n {tensor_center}\n" +
            f"Все элементы с четными индексами\n: {self.A[:, :, ::2]}"  # взял четность только в строчках
            )


class Task_1_4:
    """Задание 1.4 - Работа с формами"""
    def __init__(self):
        self.A = torch.arange(0, 24)
        print("\nЗадание 1.4\n")
    
    def __call__(self):
        tensor_1 = self.A.reshape(2, 12)
        tensor_2 = self.A.reshape(3, 8)
        tensor_3 = self.A.reshape(4, 6)
        tensor_4 = self.A.reshape(2, 3, 4)
        tensor_5 = self.A.reshape(2, 2, 2, 3)
        print(
            f"\nТензор 1:\n {tensor_1}\n" +
            f"Тензор 2:\n {tensor_2}\n" +
            f"Тензор 3:\n {tensor_3}\n" +
            f"Тензор 4:\n {tensor_4}\n" +
            f"Тензор 5:\n {tensor_5}"
            )
    

if __name__ == "__main__":
    Task_1_1.run()
    Task_1_2()()
    Task_1_3()()
    Task_1_4()()
