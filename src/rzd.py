from datetime import timedelta
from typing import Optional

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.pca import IPCA
from src.regression import SimpleLinearRegression

plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = 4, 3
plt.rcParams['grid.linestyle'] = 'dotted'
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['lines.marker'] = '.'
plt.rcParams['lines.markersize'] = 4


class PlacePriceModel:
    def __init__(self):
        self.places = None
        self.prices = None
        self.index = None
        self.columns = None

        self.ipca = IPCA(n_components=1)
        self.explained_variance = None
        self.mean = None
        self.v = None
        self.a = None

        self.regression = SimpleLinearRegression()
        self.r2 = None

    def fit(self, places: pd.DataFrame, prices: pd.DataFrame) -> 'PlacePriceModel':
        assert places.shape == prices.shape
        assert (places.columns == prices.columns).all()
        assert (places.index == prices.index).all()

        self.places = places.values
        self.prices = prices.values
        self.index = places.index
        self.columns = places.columns

        self.ipca.fit(self.places)
        self.explained_variance = self.ipca.explained_variance_ratio[0]
        self.mean = pd.Series(self.ipca.mean, index=self.columns)
        self.v = pd.Series(self.ipca.v.flatten(), index=self.columns)

        all_timestamps = np.arange(min(self.index), max(self.index) + timedelta(days=1), timedelta(days=1))
        self.a = pd.Series(self.ipca.a.flatten(), index=self.index).reindex(all_timestamps)

        self.regression.fit(self.places, self.prices)
        self.r2 = self.regression.r2

        return self

    def plot_places_model(self):
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
        fig.set_size_inches(12, 3)

        self.mean.plot(ax=ax1, c='C0')
        ax1.set_title('mean')
        ax1.set_xlabel('days until departure')

        self.v.plot(ax=ax2, c='C1')
        ax2.set_title('V(tau)')
        ax2.set_xlabel('days until departure')

        self.a.plot(ax=ax3, c='C2')
        ax3.set_title('A(t)')
        ax3.set_xlabel('date')

        fig.suptitle('Places IPCA model', fontsize=14, y=1.05)

    def plot_prices_model(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            ax = plt.gca()

        for i, dt in enumerate(self.index):
            weekday_id = dt.weekday()
            weekday = dt.strftime('%A')

            ax.scatter(self.places[i], self.prices[i], color=f'C{weekday_id}', label=weekday)

        ox = np.array([np.nanmin(self.places), np.nanmax(self.places)])
        ax.plot(ox, self.regression.b0 + self.regression.b1 * ox, c='black', ls='dashed', marker='None', zorder=5)

        handles, labels = ax.get_legend_handles_labels()
        legend = dict(zip(labels, handles))
        handles = list(legend.values())
        labels = list(legend.keys())

        ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(1, 0))

        ax.set_xlabel('places')
        ax.set_ylabel('price')
        ax.set_title('Price vs. Places regression', fontsize=14)

    def plot_residuals(self):
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        fig.set_size_inches(8, 3)

        ax1.hist(self.ipca.residuals, bins=15, color='C0')
        ax1.set_title('Places IPCA model')

        ax2.hist(self.regression.residuals, bins=15, color='C1')
        ax2.set_title('Price vs. Places regression')

        fig.suptitle('Model residuals', fontsize=14, y=1.05)
