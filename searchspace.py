import numpy as np
import logging

try:
    from pomegranate import MultivariateGaussianDistribution, GeneralMixtureModel
except ImportError:
    MultivariateGaussianDistribution = None
    GeneralMixtureModel = None

from sklearn.mixture import GaussianMixture


class SearchSpace(object):

    def reset(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def addData(self, data, score):
        return


class ExhaustiveSearch(SearchSpace):

    def __init__(self,
                 search_space=[[-3.0, 3.0], [-3.0, 3.0], [-10.0, 10.0]],
                 search_dims=[7, 7, 3]):

        x_space = np.linspace(
            search_space[0][0], search_space[0][1],
            search_dims[0])

        y_space = np.linspace(
            search_space[1][0], search_space[1][1],
            search_dims[1])

        a_space = np.linspace(
            search_space[2][0], search_space[2][1],
            search_dims[2])

        X, Y, A = np.meshgrid(x_space, y_space, a_space)  # create mesh grid

        self.search_grid = np.array([X.flatten(), Y.flatten(), A.flatten()]).T

        self.reset()

    def reset(self):
        return

    def sample(self, n=0):
        return self.search_grid


class ParticleFiltering(SearchSpace):
    def __init__(self, bnd=[1, 1, 10]):
        self.bnd = bnd
        self.reset()

    def sample(self, n=10):
        samples = []
        for i in range(n):
            if len(self.data) > 0:
                i_mean = np.random.choice(
                    list(range(len(self.data))),
                    p=self.score / np.linalg.norm(self.score, ord=1))
                sample = np.random.multivariate_normal(
                    mean=self.data[i_mean], cov=np.diag(np.array(self.bnd)))
            else:
                sample = np.random.multivariate_normal(
                    mean=np.zeros(len(self.bnd)),
                    cov=np.diag(np.array(self.bnd) * 3))

            samples.append(sample)
        return np.array(samples)

    def addData(self, data, score):
        score = score.clip(min=1e-5)  # prevent sum=0 in case of bad scores
        self.data = data
        self.score = score

    def reset(self):
        if len(self.bnd) == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.ones(np.shape(self.data)[0])
        self.score = self.score / np.linalg.norm(self.score, ord=1)


class KalmanFiltering(SearchSpace):
    def __init__(self, bnd=[1, 1, 10]):
        self.bnd = bnd
        self.reset()

    def sample(self, n=10):
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def addData(self, data, score):
        score = score.clip(min=1e-5)  # prevent sum=0 in case of bad scores
        self.data = np.concatenate((self.data, data))
        self.score = np.concatenate((self.score, score))
        self.mean = np.average(self.data, weights=self.score, axis=0)
        self.cov = np.cov(self.data.T, ddof=0, aweights=self.score)

    def reset(self):
        self.mean = np.zeros(len(self.bnd))
        self.cov = np.diag(self.bnd)
        if len(self.bnd) == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.array([])


class GaussianMixtureModel(SearchSpace):

    def __init__(self, n_comp=5, dim=3):
        self.dim = dim
        self.reset(n_comp)

    def _sample_gaussian(self, mean, cov, n):
        n = max(int(n), 1)
        return np.atleast_2d(
            np.random.multivariate_normal(np.asarray(mean), np.asarray(cov), size=n))

    def _set_default_model(self, mean, cov):
        self.model = None
        self.model_mean = np.asarray(mean)
        self.model_cov = np.asarray(cov)
        self.model_backend = "gaussian"

    def _sample_model(self, n):
        n = max(int(n), 1)
        if self.model_backend == "pomegranate" and self.model is not None:
            return np.stack(self.model.sample(n))
        if self.model_backend == "sklearn" and self.model is not None:
            return self.model.sample(n)[0]
        return self._sample_gaussian(self.model_mean, self.model_cov, n)

    def sample(self, n=10):
        try:
            X1 = self._sample_model(int(np.round(0.8 * n)))
            if self.dim == 2:
                mean = np.mean(X1, axis=0)
                std = np.diag([1.0, 1.0])
                X2 = self._sample_gaussian(mean, std, int(np.round(0.1 * n)))

                mean = np.mean(X1, axis=0)
                std = np.diag([1e-3, 1e-3])
                X3 = self._sample_gaussian(mean, std, int(np.round(0.1 * n)))

            else:
                mean = np.mean(X1, axis=0)
                std = np.diag([1.0, 1.0, 1e-3])
                X2 = self._sample_gaussian(mean, std, int(np.round(0.1 * n)))

                mean = np.mean(X1, axis=0)
                std = np.diag([1e-3, 1e-3, 10.0])
                X3 = self._sample_gaussian(mean, std, int(np.round(0.1 * n)))

            X = np.concatenate((X1, X2, X3))

        except ValueError:
            logging.info("exception caught on sampling")
            if self.dim == 2:
                mean = np.zeros(self.dim)
                std = np.diag([1.0, 1.0])
                X = self._sample_gaussian(mean, std, int(n))
            else:
                mean = np.zeros(self.dim)
                std = np.diag([1.0, 1.0, 5.0])
                X = self._sample_gaussian(mean, std, int(n))
        return X

    def addData(self, data, score):
        score = score.clip(min=1e-5)
        self.data = data
        self.score = score

        score_normed = self.score / np.linalg.norm(self.score, ord=1)
        if len(self.data) == 0:
            return

        if (GeneralMixtureModel is not None and
                MultivariateGaussianDistribution is not None):
            try:
                model = GeneralMixtureModel.from_samples(
                    MultivariateGaussianDistribution,
                    n_components=self.n_comp,
                    X=self.data,
                    weights=score_normed)
                self.model = model
                self.model_backend = "pomegranate"
                return
            except Exception:
                logging.info("failed to fit pomegranate GMM, falling back to sklearn")

        try:
            sample_count = max(len(self.data), self.n_comp * 20)
            indices = np.random.choice(
                len(self.data), size=sample_count, replace=True, p=score_normed)
            sampled_data = self.data[indices]
            model = GaussianMixture(
                n_components=min(self.n_comp, len(sampled_data)),
                covariance_type='full',
                reg_covar=1e-6,
                random_state=0)
            model.fit(sampled_data)
            self.model = model
            self.model_backend = "sklearn"
        except Exception:
            logging.info("catched an exception")

    def reset(self, n_comp=5):
        self.n_comp = n_comp

        if self.dim == 2:
            self.data = np.array([[], []]).T
        else:
            self.data = np.array([[], [], []]).T
        self.score = np.ones(np.shape(self.data)[0])
        self.score = self.score / np.linalg.norm(self.score, ord=1)
        if self.dim == 2:
            self._set_default_model(np.zeros(self.dim), np.diag([1.0, 1.0]))
        else:
            self._set_default_model(np.zeros(self.dim), np.diag([1.0, 1.0, 5.0]))
