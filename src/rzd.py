from datetime import timedelta

import pandas as pd
import numpy as np

from src.pca import ImputePCA
from src.regression import SimpleLinearRegression


class PlacePriceModel:
    def __init__(self):
        self.places = None
        self.prices = None
        self.index = None
        self.columns = None

        self.impute_pca = None
        self.explained_variance = None
        self.mean = None
        self.v = None
        self.a = None

        self.regression = None
        self.r2 = None

    def fit(self, places: pd.DataFrame, prices: pd.DataFrame) -> 'PlacePriceModel':
        assert places.shape == prices.shape
        assert (places.columns == prices.columns).all()
        assert (places.index == prices.index).all()

        self.places = places.values
        self.prices = prices.values
        self.index = places.index
        self.columns = places.columns
        
        self.impute_pca = ImputePCA().fit(self.places)
        self.explained_variance = self.impute_pca.explained_variance_ratio[0]
        self.mean = pd.Series(self.impute_pca.mean, index=self.columns)
        self.v = pd.Series(self.impute_pca.v_norm[:, 0], index=self.columns)

        all_timestamps = np.arange(min(self.index), max(self.index) + timedelta(days=1), timedelta(days=1))
        self.a = pd.Series(self.impute_pca.a_norm[:, 0], index=self.index).reindex(all_timestamps)

        self.regression = SimpleLinearRegression()
        self.regression.fit(self.places, self.prices)
        self.r2 = self.regression.r2

        return self

    # def plot_places_model(self):
    #     fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    #     fig.set_size_inches(12, 3)
    #
    #     self.mean.plot(ax=ax1, c='C0')
    #     ax1.set_title('mean')
    #     ax1.set_xlabel('days until departure')
    #
    #     self.v.plot(ax=ax2, c='C1')
    #     ax2.set_title('V(tau)')
    #     ax2.set_xlabel('days until departure')
    #
    #     self.a.plot(ax=ax3, c='C2')
    #     ax3.set_title('A(t)')
    #     ax3.set_xlabel('date')
    #
    #     fig.suptitle('Places IPCA model', fontsize=14, y=1.05)
    #
    # def plot_prices_model(self, ax: Optional[plt.Axes] = None):
    #     if ax is None:
    #         ax = plt.gca()
    #
    #     for i, dt in enumerate(self.index):
    #         weekday_id = dt.weekday()
    #         weekday = dt.strftime('%A')
    #
    #         ax.scatter(self.places[i], self.prices[i], color=f'C{weekday_id}', label=weekday)
    #
    #     ox = np.array([np.nanmin(self.places), np.nanmax(self.places)])
    #     ax.plot(ox, self.regression.b0 + self.regression.b1 * ox, c='black', ls='dashed', marker='None', zorder=5)
    #
    #     handles, labels = ax.get_legend_handles_labels()
    #     legend = dict(zip(labels, handles))
    #     handles = list(legend.values())
    #     labels = list(legend.keys())
    #
    #     ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(1, 0))
    #
    #     ax.set_xlabel('places')
    #     ax.set_ylabel('price')
    #     ax.set_title('Price vs. Places regression', fontsize=14)
    #
    # def plot_residuals(self):
    #     fig, (ax1, ax2) = plt.subplots(ncols=2)
    #     fig.set_size_inches(8, 3)
    #
    #     ax1.hist(self.ipca.residuals, bins=15, color='C0')
    #     ax1.set_title('Places IPCA model')
    #
    #     ax2.hist(self.regression.residuals, bins=15, color='C1')
    #     ax2.set_title('Price vs. Places regression')
    #
    #     fig.suptitle('Model residuals', fontsize=14, y=1.05)
