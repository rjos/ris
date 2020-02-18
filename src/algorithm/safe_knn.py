from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor, KNeighborsClassifier
import logging

logger = logging.getLogger(__name__)


class SafeBase:
    def __init__(self, model, *args, **kwargs):
        if args:
            self.init_n = args[0]
        else:
            self.init_n = kwargs.get('n_neighbors', 5)

        self.current_n = self.init_n
        self.knn = model(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.refit = False
        self.model = model

    def _resize_model(self, new_n):
        if self.args:
            tmp = list(self.args)
            tmp[0] = new_n
            self.args = tuple(tmp)
        else:
            self.kwargs['n_neighbors'] = new_n

        logger.debug('Refitting model %d from %d to %d', self.init_n, self.current_n, new_n)
        self.knn = self.model(*self.args, **self.kwargs)
        self.current_n = new_n

    def fit(self, x, *y):

        n = x.shape[0]
        if n < self.current_n:
            self._resize_model(n)
            self.refit = True
        elif self.refit and n > self.current_n:
            if n >= self.init_n:
                n = self.init_n
                self.refit = False

            self._resize_model(n)

        return self.knn.fit(x, *y)

    def __getattr__(self, item):
        return getattr(self.knn, item)


class SafeKnnRegressor(SafeBase):
    def __init__(self, *args, **kwargs):
        super(SafeKnnRegressor, self).__init__(KNeighborsRegressor, *args, **kwargs)


class SafeNearestNeighbors(SafeBase):
    def __init__(self, *args, **kwargs):
        super(SafeNearestNeighbors, self).__init__(NearestNeighbors, *args, **kwargs)


class SafeKnnClassifier(SafeBase):
    def __init__(self, *args, **kwargs):
        super(SafeKnnClassifier, self).__init__(KNeighborsClassifier, *args, **kwargs)
