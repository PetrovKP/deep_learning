import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from scipy.special import xlogy

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100), solver='adagrad',
                 batch_size=1, learning_rate=0.001, momentum=0.9, eps=1e-8,
                 max_iter=200, random_state=32, shuffle=True, verbose=False):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.momentum = momentum
        self.eps = eps
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.shuffle = shuffle
        self.verbose = verbose

    def __stable_softmax(self, x):
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        return x / x.sum(axis=1, keepdims=True)

    def __crossentropy_loss(self, y_true, y_prob):
        return - xlogy(y_true, y_prob).sum()

    def __forward_layer(self, x, w, activation_function):
        out = np.dot(x, w)
        if activation_function is not None:
            out = activation_function(out)
        return out

    def __forward_propagate(self, x):
        weights = self.weights
        out_activations = [x]
        for weight, activataion in zip(weights, self.functions):
            out = self.__forward_layer(out_activations[-1], weight, activataion)
            out_activations.append(out)
        return out_activations

    def __back_propagation(self, activations, y):
        weights = self.weights
        coef_grads = [np.empty_like(a_layer) for a_layer in weights]

        deltas = activations[-1] - y
        coef_grads[-1] = np.dot(activations[-2].T, deltas)

        for i in range(len(weights)-2, -1, -1):
            deltas =  np.dot(deltas, weights[i + 1].T)
            coef_grads[i] = np.dot(activations[i].T, deltas)

        return coef_grads

    def __init_layer(self, input_size, output_size):
        a = 2.0/(input_size + output_size)
        w = np.random.uniform(-a, a, (input_size, output_size))
        return w

    def fit(self, X, y):
        np.random.seed(self.random_state)

        self._label_binarizer = LabelBinarizer()
        y_train = y
        X_train = X
        y = self._label_binarizer.fit_transform(y)
        self._num_classes = len(self._label_binarizer.classes_)

        n, p = X.shape
        s = self.hidden_layer_sizes[0]

        self.weights = [
            self.__init_layer(p, s),
            self.__init_layer(s, self._num_classes)
        ]

        self.functions = [
            None,
            self.__stable_softmax,
        ]

        accum_grad = [np.zeros_like(param) for param in self.weights]

        for j in range(self.max_iter):
            accumulated_loss = 0.0

            if self.shuffle:
                indices = np.arange(n)
                np.random.shuffle(indices)
                X = X.take(indices, axis=0)
                y = y.take(indices, axis=0)

            for i in range(0, n, self.batch_size):
                X_batch = X[i : i + self.batch_size]
                y_batch = y[i : i + self.batch_size]

                activations = self.__forward_propagate(X_batch)

                y_prob = activations[-1]

                accumulated_loss += self.__crossentropy_loss(y_batch, y_prob)
                coef_grads = self.__back_propagation(activations, y_batch)

                coef_grads = [grad / self.batch_size for grad in coef_grads]
                accum_grad = [accum + grad**2 for accum, grad in zip(accum_grad, coef_grads)]
                inv_accum_grad = [self.learning_rate / np.sqrt(self.eps + accum) for accum in accum_grad]
                self.weights = [weight - inv_accum * grad for weight, inv_accum, grad in zip(self.weights, inv_accum_grad, coef_grads)]

            if self.verbose:
                loss = accumulated_loss / X.shape[0]
                y_pred = self.predict(X_train)
                accuracy = (y_pred == y_train).mean()
                print("Epoch {}/{};\t Train accuracy: {:.3f} \t Loss : {:.3f}".format(j + 1, self.max_iter, accuracy, loss))

        return self

    def predict(self, X):
        activations = self.__forward_propagate(X)
        y_pred = activations[-1]
        return self._label_binarizer.inverse_transform(y_pred)

from sklearn.datasets import fetch_mldata

data_train = pd.read_csv("../dataset/mldata/mnist_train.csv", header=None)
data_test = pd.read_csv("../dataset/mldata/mnist_test.csv", header=None)

x_train = np.ascontiguousarray(data_train[data_train.columns[:-1]].values, dtype=np.float32)
y_train = np.ascontiguousarray(data_train[data_train.columns[-1]].values, dtype=np.float32)
x_test = np.ascontiguousarray(data_test[data_test.columns[:-1]].values, dtype=np.float32)
y_test = np.ascontiguousarray(data_test[data_test.columns[-1]].values, dtype=np.float32)

print('train size: ', x_train.shape, y_train.shape)
print('test size: ', x_test.shape, y_test.shape)

x_train /= 255
x_test /= 255

# ## Grid search parameters for Neural Network

from sklearn.model_selection import GridSearchCV

parameters = {
    'hidden_layer_sizes': [(16,), (32,), (48,), (64,), (76,), (92,)],
}

estimator = DNNClassifier(solver='adagrad',
     batch_size=256, learning_rate=0.05, max_iter=50,
     random_state=777, verbose=False)

clf = GridSearchCV(estimator, parameters, cv=5, scoring='accuracy')
clf.fit(x_train, y_train)
print("Лучший подбор параметра для DNNClassifier: {}".format(clf.best_params_))
print("Лучший scope для DNNClassifier: {}".format(clf.best_score_))

pd.DataFrame(clf.cv_results_).plot(x='param_hidden_layer_sizes', y=[ 'mean_test_score', 'mean_train_score'])

best_estimator = clf.best_estimator_
up_params = {'verbose': True, 'max_iter': 30}
best_estimator.set_params(**up_params)

y_pred = best_estimator.predict(x_test)
print((y_pred == y_test).mean())


# ## Versus the sklearn MLP

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=30, alpha=0, activation='identity',
                     batch_size=256, solver='adam', verbose=10, random_state=777,
                     tol=0, shuffle=True)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print((y_pred == y_test).mean())
