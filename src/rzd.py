from datetime import timedelta
from typing import Optional

import pandas as pd
import numpy as np

from src.pca import ImputePCA
from src.regression import SimpleLinearRegression


class PlacePriceModel:
    """Модель данных РЖД (места и цены).
    
    Использует `ImputePCA` для построения модели ГК на данных
    о местах и `SimpleLinearRegression` для построения лин. регрессии
    цены в зависимости от числа мест.
    """
    
    def __init__(self):
        self.places = None  # матрица с местами
        self.prices = None  # матрица с ценами
        self.index = None  # индексы (даты)
        self.columns = None  # столбцы (число дней до отправления)
        
        self.impute_pca = None  # модель PCA
        self.explained_variance = None  # объяснённая дисперсия 1ГК
        self.mu = None  # вектор средних
        self.v = None  # с.в., умноженный на СКО 1ГК
        self.a = None  # 1ГК с СКО=1

        self.regression = None  # модель регрессии
        self.r2 = None  # её коэф. детерминации

    def fit(self, places: pd.DataFrame, prices: pd.DataFrame) -> 'PlacePriceModel':
        assert places.shape == prices.shape
        assert (places.columns == prices.columns).all()
        assert (places.index == prices.index).all()

        self.places = places.values
        self.prices = prices.values
        self.index = places.index
        self.columns = places.columns
        
        self.impute_pca = ImputePCA(n_components=1)
        self.impute_pca.fit_predict(self.places)

        self.explained_variance = self.impute_pca.explained_variance_ratio[0]
        self.mu = pd.Series(self.impute_pca.mean_vector, index=self.columns)
        self.v = pd.Series(self.impute_pca.loadings[:, 0], index=self.columns)

        all_timestamps = np.arange(min(self.index), max(self.index) + timedelta(days=1), timedelta(days=1))
        self.a = pd.Series(self.impute_pca.scaled_components[:, 0], index=self.index).reindex(all_timestamps)

        self.regression = SimpleLinearRegression()
        self.regression.fit(self.places, self.prices)
        self.r2 = self.regression.r2

        return self
