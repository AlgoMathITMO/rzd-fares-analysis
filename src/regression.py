import numpy as np


class SimpleLinearRegression:
    """Тупо парная линейная регрессия, но с возможностью
    подавать на вход многомерные матрицы.
    
    Обрабатывает пропуски, т.е. обучается только на тех позициях,
    на которых и `x`, и `y` имеют значения.
    """
    
    def __init__(self):
        self.b0 = None  # сдвиг
        self.b1 = None  # наклон

        self.residuals = None  # остатки
        self.r2 = None  # коэф. детерминации

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SimpleLinearRegression':
        assert x.shape == y.shape

        x = x.flatten()
        y = y.flatten()

        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        self.b1 = np.cov(x, y)[0, 1] / x.var()
        self.b0 = y.mean() - self.b1 * x.mean()

        z = self.b0 + self.b1 * x
        self.residuals = y - z
        self.r2 = 1 - self.residuals.var() / y.var()

        return self
