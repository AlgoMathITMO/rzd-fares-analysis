import numpy as np


class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

        self.mean = None
        self.eigval = None
        self.eigvec = None

        self.explained_variance_ratio = None

        self.v = None
        self.a = None

        self.reconstructed = None
        self.residuals = None

    def fit(self, x: np.ndarray) -> 'PCA':
        self.mean = x.mean(axis=0)
        x_centered = x - self.mean

        cov = np.cov(x_centered.T)

        self.eigval, self.eigvec = np.linalg.eig(cov)

        ids = np.argsort(self.eigval)[::-1]
        self.eigval = self.eigval[ids]
        self.eigvec = self.eigvec[:, ids]

        self.explained_variance_ratio = (self.eigval / self.eigval.sum()).cumsum()

        self.v = self.eigvec[:, :self.n_components]
        self.v *= self.v[0] / np.abs(self.v[0])

        self.a = x_centered.dot(self.v)

        self.reconstructed = self.mean + self.a.dot(self.v.T)
        self.residuals = (x - self.reconstructed).flatten()

        return self


class IPCA:
    def __init__(self, n_components: int = 2, maxiter: int = 500, tol: float = 1e-5):
        self.n_components = n_components
        self.maxiter = maxiter
        self.tol = tol

        self.mean = None
        self.eigval = None
        self.eigvec = None

        self.w = None
        self.x_filled = None

        self.explained_variance_ratio = None

        self.v = None
        self.a = None

        self.reconstructed = None
        self.residuals = None

        self.scores = None

    def step(self):
        self.mean = self.x_filled.mean(axis=0)
        x_centered = self.x_filled - self.mean

        cov = np.cov(x_centered.T)

        self.eigval, self.eigvec = np.linalg.eigh(cov)

        ids = np.argsort(self.eigval)[::-1]
        self.eigval = self.eigval[ids]
        self.eigvec = self.eigvec[:, ids]

        self.explained_variance_ratio = (self.eigval / self.eigval.sum()).cumsum()[:self.n_components]

        self.v = self.eigvec[:, :self.n_components]

        for i in range(self.v.shape[1]):
            if (self.v[:, i] < 0).mean() > 0.5:
                self.v[:, i] *= -1

        self.a = x_centered.dot(self.v)

        s2 = self.eigval[self.n_components:].mean()
        diag = np.diag((self.eigval[:self.n_components] - s2) / self.eigval[:self.n_components])

        new = self.mean + self.a.dot(diag).dot(self.v.T)
        self.x_filled = self.x_filled * self.w + new * (1 - self.w)

        self.reconstructed = self.mean + self.a.dot(self.v.T)

        self.residuals = (self.x_filled - self.reconstructed)[self.w.astype(bool)]

    def fit(self, x: np.ndarray) -> 'IPCA':
        self.x_filled = x.copy()

        missing = np.where(np.isnan(self.x_filled))

        self.w = np.ones(self.x_filled.shape)
        self.w[missing] = 0

        mean = np.nanmean(self.x_filled, axis=0)

        if np.isnan(mean).any():
            raise RuntimeError('fully nan columns in fitted data are not accepted')

        self.x_filled[missing] = mean[missing[1]]

        self.scores = []

        for _ in range(self.maxiter):
            a_old = self.a
            self.step()

            if a_old is not None:
                score = ((self.a - a_old) ** 2).sum() ** 0.5
                self.scores.append(score)

                if score <= self.tol:
                    return self

        msg = f'did not converge in {self.maxiter} iterations. best score: {min(self.scores)} > {self.tol}'
        raise RuntimeError(msg)
