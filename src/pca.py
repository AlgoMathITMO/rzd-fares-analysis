from typing import Optional

import numpy as np

from src.missing_values import impute_average


class ImputePCA:
    """Метод главных компонент с заполнением пропусков
    (см. `impute_average`).
    """
    
    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

        self.explained_variance_ratio = None
        
        self.mean_vector = None  # вектор средних
        
        self.eigenvectors = None  # собственные векторы
        self.loadings = None  # с.в., умноженные на СКО соответствующих ГК
        
        self.components = None  # ГК
        self.scaled_components = None  # ГК с СКО=1
        
        self.mse = None  # MSE при восстановлении данных с помощью ГК

    def fit_predict(self, x: np.ndarray, return_scaled: bool = True) -> np.ndarray:
        """В целом, стандартный фит для PCA, но:
        
        1. заполняет пропуски,
        2. пытается подобрать консистентный знак для собст. векторов
        (в РЖД-шных данных, например, с.в. должны быть преимущественно
        положительными).
        
        Если `return_scaled=True`, возвращает нормированные ГК (т.е. СКО=1).
        """
        
        n_components = self.n_components or x.shape[1]
        
        x_filled = impute_average(x)

        self.mean_vector = x_filled.mean(axis=0)
        x_centered = x_filled - self.mean_vector

        cov = np.cov(x_centered.T)

        eigval, eigvec = np.linalg.eigh(cov)

        idx = np.argsort(eigval)[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]

        self.explained_variance_ratio = (eigval / eigval.sum()).cumsum()

        self.eigenvectors = eigvec[:, :n_components]

        for i in range(self.eigenvectors.shape[1]):
            if (self.eigenvectors[:, i] < 0).mean() > 0.5:
                self.eigenvectors[:, i] *= -1
                
        self.components = x_centered.dot(self.eigenvectors)

        std = self.components.std(axis=0, ddof=1, keepdims=True)
        self.scaled_components = self.components / std
        self.loadings = self.eigenvectors * std
        
        reconstructed = self.reconstruct(self.components, scaled=False)
        res = x - reconstructed
        res = res[~np.isnan(res)]
        self.mse = (res ** 2).mean() ** 0.5
        
        if return_scaled:
            return self.scaled_components
        else:
            return self.components
    
    def reconstruct(self, components: np.ndarray, scaled: bool = True) -> np.ndarray:
        """Восстановление оригинальных данных с помощью ГК.
        Если `scaled=True`, ожидает, что в `components` подаются
        нормированные ГК (т.е. СКО=1).
        """
        
        if scaled:
            return self.mean_vector + components.dot(self.loadings.T)
        else:
            return self.mean_vector + components.dot(self.eigenvectors.T)
    

# class IPCA:
#     def __init__(self, n_components: int = 10, maxiter: int = 10000, tol: float = 1e-4):
#         self.n_components = n_components
#         self.maxiter = maxiter
#         self.tol = tol

#         self.mean = None
#         self.eigval = None
#         self.eigvec = None

#         self.w = None
#         self.x_filled = None

#         self.explained_variance_ratio = None

#         self.v = None
#         self.a = None
        
#         self.v_norm = None
#         self.a_norm = None

#         self.reconstructed = None
#         self.residuals = None

#         self.scores = None

#     def step(self):
#         self.mean = self.x_filled.mean(axis=0)
#         x_centered = self.x_filled - self.mean

#         cov = np.cov(x_centered.T)

#         self.eigval, self.eigvec = np.linalg.eigh(cov)

#         ids = np.argsort(self.eigval)[::-1]
#         self.eigval = self.eigval[ids]
#         self.eigvec = self.eigvec[:, ids]

#         self.explained_variance_ratio = (self.eigval / self.eigval.sum()).cumsum()[:self.n_components]

#         self.v = self.eigvec[:, :self.n_components]

#         for i in range(self.v.shape[1]):
#             if (self.v[:, i] < 0).mean() > 0.5:
#                 self.v[:, i] *= -1

#         self.a = x_centered.dot(self.v)
        
#         s2 = self.eigval[self.n_components:].mean()
#         diag = np.diag((self.eigval[:self.n_components] - s2) / self.eigval[:self.n_components])

#         new = self.mean + self.a.dot(diag).dot(self.v.T)
#         self.x_filled = self.x_filled * self.w + new * (1 - self.w)

#         self.reconstructed = self.mean + self.a.dot(self.v.T)

#         self.residuals = (self.x_filled - self.reconstructed)[self.w.astype(bool)]

#     def fit(self, x: np.ndarray, min_periods: int = 10) -> 'IPCA':
#         self.x_filled = x.copy()

#         missing = np.where(np.isnan(self.x_filled))

#         self.w = np.ones(self.x_filled.shape)
#         self.w[missing] = 0
        
#         if (self.w.sum(axis=0) < min_periods).any():
#             raise RuntimeError(f'some columns have less than {min_periods} values')
            
#         if (self.w.sum(axis=1) < min_periods).any():
#             raise RuntimeError(f'some rows have less than {min_periods} values')

#         mean = np.nanmean(self.x_filled, axis=0)

#         self.x_filled[missing] = mean[missing[1]]

#         self.scores = []

#         for _ in range(self.maxiter):
#             a_old = self.a
#             self.step()

#             if a_old is None:  #first iteration
#                 continue

#             score = (((self.a - a_old) ** 2).sum(axis=1) ** 0.5).mean()
#             self.scores.append(score)

#             if score <= self.tol:
#                 std = self.a.std(axis=0, ddof=1, keepdims=True)
#                 self.a_norm = self.a / std
#                 self.v_norm = self.v * std

#                 return self

#         msg = f'did not converge in {self.maxiter} iterations. best score: {min(self.scores)} > {self.tol}'
#         raise RuntimeError(msg)
