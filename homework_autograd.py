import torch


class Task_2_1:
    """Задание 2.1 - Простые вычисления с градиентами"""
    def __init__(self):
        """Создаем простые скалярные тензоры для упрощения проверки результата"""
        self.x = torch.tensor(1.0, requires_grad=True)
        self.y = torch.tensor(3.0, requires_grad=True)
        self.z = torch.tensor(5.0,  requires_grad=True)
        print(
            f"\nЗадание 2.1\n" +
            f"X: {self.x}\n" +
            f"Y: {self.y}\n" +
            f"Z: {self.z}"
            )

    def _func(self) -> torch.Tensor:
        """Метод для расчета значения функции"""
        return self.x**2 + self.y**2 + self.z**2 + 2*self.x*self.y*self.z

    def _grad(self):
        """Метод, выводящий в консоль значения градиентов"""
        f_x = self._func().backward()
        print(
            f"\nГрадиент X:\n {self.x.grad}\n" +  # f'x = 2*x + 2*y*z = 32
            f"Градиент Y:\n {self.y.grad}\n" +  # f'y = 2*y +2*x*z = 16
            f"Градиент Z:\n {self.z.grad}\n"  # f'z = 2*z + 2*x*y = 16
            )

    def __call__(self):
        self._grad()


class Task_2_2:
    """Задача 2.2 - Градиент функции потерь"""
    def __init__(self):
        """Создаем абстрактные входные данные и ожидаемый результат"""
        self.x = torch.tensor([1.0, 3.0, 5.0, 10.0, 20.0]) 
        self.y_true = torch.tensor([3.0, 0.0, 3.5, 5.2, 2.28]) 
        self.w = torch.tensor(0.0,  requires_grad=True)
        self.b = torch.tensor(0.0,  requires_grad=True)
        print(
            f"\nЗадание 2.2\n" +
            f"X: {self.x}\n" +
            f"Y_true: {self.y_true}"
            )

    def _func(self) -> torch.Tensor:
        """Метод для расчета y_pred"""
        return self.w * self.x + self.b

    def _grad(self):
        """Метод, выводящий в консоль значения градиентов и функции ошибки"""
        y_pred = self._func()
        MSE = torch.mean((y_pred - self.y_true)**2)
        MSE.backward()
        print(
            f"\nMSE: {MSE:.4f}\n" +
            f"Градиент w:\n {self.w.grad:.4f}\n" +  
            f"Градиент b:\n {self.b.grad:.4f}"
            )

    def __call__(self):
        self._grad()


class Task_2_3:
    """Задание 2.3 - Цепное правило"""
    def __init__(self):
        """Создаем скалярный тензор x"""
        self.x = torch.tensor(3.0, requires_grad=True) 
        print(
            f"\nЗадание 2.3\n" +
            f"X: {self.x}"
        )

    def _func(self) -> torch.Tensor:
        """Метод для расчета f(x)"""
        return torch.sin(self.x**2 + 1)

    def _grad(self):
        """Метод, выводящий в консоль значения градиентов"""
        f_x = self._func()
        f_x.backward()
        grad_autograd = torch.autograd.grad(torch.sin(self.x**2 + 1), self.x)[0]
        print(
            f"\nF(x): {f_x:.4f}\n" +
            f"Градиент F'x:\n {self.x.grad:.4f}\n" +
            f"Градиент F'x через autograd.grad:\n {grad_autograd:.4f}"
            )

    def __call__(self):
        self._grad()


if __name__ == "__main__":
    Task_2_1()()
    Task_2_2()()
    Task_2_3()()
