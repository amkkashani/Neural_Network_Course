import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # ****** Perceptron ******
    t1_perceptron("dataset1.csv")

    #    ****** Adaline ******
    t2_adaline_normal_form("dataset1.csv")

    #  ****** Adaline for non-linear  *******
    t3_adalin_curve_form("dataset3.csv")


def t1_perceptron(dir):
    data = pd.read_csv(dir)

    test = data.sample(frac=0.2)
    train = data.drop(test.index)

    X = train[['x', 'y']].to_numpy()
    y = train['label'].to_numpy()

    test_x = test[['x', 'y']].to_numpy()
    test_y = test['label'].to_numpy()

    blues = data[data['label'] == 0]
    reds = data[data['label'] == 1]

    perceptron = Perceptron(2)
    perceptron.train_by_data(X, y, 100, test_X=test_x, test_y=test_y)

    precision = perceptron.precision_calculate(X, y)
    print(perceptron.W)
    print(f'precision {precision}')

    perceptron.plot_boundary_and_data(blues, reds, precision)
    perceptron.plot_accuracy_history(precision)


def t2_adaline_normal_form(dir):
    data = pd.read_csv(dir)
    blues = data[data['label'] == 0]
    reds = data[data['label'] == 1]

    test = data.sample(frac=0.2)
    train = data.drop(test.index)
    X = train[['x', 'y']].to_numpy()
    y = train['label'].to_numpy()

    test_x = test[['x', 'y']].to_numpy()
    test_y = test['label'].to_numpy()


    perceptron = Adalin(2, learning_rate=0.005)
    perceptron.train_by_data(X, y, epoch=35, test_X=test_x, test_y=test_y)

    precision = perceptron.precision_calculate(X, y)
    print(perceptron.W)
    print(f'precision {precision}')
    perceptron.plot_boundary_and_data(blues, reds, precision)
    perceptron.plot_accuracy_history(precision)


def t3_adalin_curve_form(dir="dataset3.csv"):
    data = pd.read_csv(dir)

    blues = data[data['label'] == 0]
    reds = data[data['label'] == 1]

    test = data.sample(frac=0.2)
    train = data.drop(test.index)

    X = train[['x', 'y']].to_numpy()
    y = train['label'].to_numpy()

    test_x = test[['x', 'y']].to_numpy()
    test_y = test['label'].to_numpy()

    X = convert_x_train_to_multi_dimension(X)
    test_x = convert_x_train_to_multi_dimension(test_x)

    adalin = Adalin(5, learning_rate=0.01)
    adalin.train_by_data(X, y, epoch=50, test_X=test_x, test_y=test_y)

    precision = adalin.precision_calculate(X, y)
    print(adalin.W)
    print(f'precision {precision}')
    adalin.plot_curve_boundary_and_data(blues, reds, precision)
    adalin.plot_accuracy_history(precision)


def convert_x_train_to_multi_dimension(X):
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()  # convert pandas to numpy
    x0 = X[:, 0]
    x1 = X[:, 1]

    return np.array([x0, x1, np.power(x0, 2), np.power(x1, 2), np.multiply(x0, x1)]).T


def sigmoid(x):
    print(1 / (1 + np.exp(-x)))
    return 1 / (1 + np.exp(-x))


def step_function(x):
    if x > 0.5:
        return 1
    else:
        return 0


def bipolar_step_function(x):  # designed for ADALINE
    if x > 0:
        return 1
    else:
        return -1


class Perceptron:
    def __init__(self, degree, activation_functions=step_function):  # initial w , and b
        self.degree = degree
        self.W = np.random.rand(degree + 1) * 2 - np.ones(degree + 1)  # between (-1 and 1]   && +1 for bias
        self.activation_functions = activation_functions
        self.history_err_train = []
        self.history_err_test = []

    def train_by_data(self, X, y, epoch, test_X, test_y):
        ##
        # X is matrix of inputs
        # y is matrix of results
        ##
        for i in range(epoch):
            prev_W = self.W
            for j in range(X.shape[0]):
                x = X[j, :]
                y_exact = y[j]
                self.train_a_data(x, y_exact)

            # calculate precision of train set and test set
            self.history_err_train.append(self.precision_calculate(X, y))
            self.history_err_test.append(self.precision_calculate(test_X, test_y))

            if (self.W == prev_W).all():
                print("line has been founded mission accomplished")
                return

    def train_a_data(self, X, y):  # just for classification
        ###
        # X is a vector
        # y is a result value
        ###
        prediction = self.calculate_value(X)
        X = np.append(np.ones(1), X)

        # in this specific situation  t-y is 1 or -1
        if y == 1 and prediction != 1:
            self.W = self.W + X
        elif y == 0 and prediction != 0:
            self.W = self.W - X

    def calculate_value(self, X):
        if not isinstance(X, np.ndarray):  # not necessary and can remove
            print("its not a numpy matrix")
            return

        X = np.append(np.ones(1), X)
        z = np.dot(X, self.W)
        result = self.activation_functions(z)

        if result > 0.5:
            return 1
        else:
            return 0

    def precision_calculate(self, X, y):
        ##
        # X matrix
        # y vector
        ##
        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            print("precision calculator have a problem")
            return None
        m = X.shape[0]
        corrects = 0
        for i in range(m):
            if self.calculate_value(X[i]) == y[i]:
                corrects += 1

        return corrects / m

    def plot_boundary_and_data(self, blues, reds, precision):
        ##
        # blues are pandas array
        # reds are pandas array
        ##
        x_axis = np.arange(-3, 8, 0.03)
        w = self.W
        y_axis = -w[0] / w[2] - x_axis * w[1] / w[2]

        plt.title(f'Perceptron \nprecision : {precision}')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(x_axis, y_axis)
        plt.scatter(blues['x'], blues['y'])
        plt.scatter(reds['x'], reds['y'])
        plt.show()

    def plot_accuracy_history(self, precision):
        x = np.arange(0, len(self.history_err_test), 1)
        plt.title(f'Perceptron \n train_precision : {self.history_err_train[-1]}   test_precision : {self.history_err_test[-1]} ')
        plt.xlabel("epoch")
        plt.ylabel("precision")
        plt.plot(x, self.history_err_train, label="train", alpha=0.5)
        plt.plot(x, self.history_err_test, label="test", alpha=0.5)
        plt.legend()
        plt.show()


class Adalin:
    def __init__(self, degree, learning_rate, activation_functions=bipolar_step_function):  # initial w , and b
        self.degree = degree
        self.W = np.random.rand(degree + 1) * 2 - np.ones(degree + 1)  # between (-1 and 1]   && +1 for bias
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.history_err_train = []
        self.history_err_test = []
        self.initial_learning_rate = learning_rate

    def train_by_data(self, X, y, epoch, test_X, test_y):
        ##
        # X is matrix of inputs
        # y is matrix of results
        ##
        for i in range(epoch):
            self.learning_rate -= self.learning_rate * (i / (epoch * 1.1))
            prev_W = self.W
            for j in range(X.shape[0]):
                x = X[j, :]
                y_exact = y[j]
                self.train_a_data(x, y_exact)

            # calculate precision of train set and test set
            self.history_err_train.append(self.precision_calculate(X, y))
            self.history_err_test.append(self.precision_calculate(test_X, test_y))

            # if (self.W == prev_W).all():
            #     print("line has been founded mission accomplished")
            #     return

    def train_a_data(self, X, y):  # just for classification
        ###
        # X is a vector
        # y is a result value
        ###
        prediction = self.calculate_value_without_activision_function(X)
        X = np.append(np.ones(1), X)

        # adaline requirement  is a polar output for NN
        if y == 0:
            y = -1

        self.W += self.learning_rate * (y - prediction) * X

    def calculate_value_without_activision_function(self, X):
        if not isinstance(X, np.ndarray):  # not necessary and can remove
            print("its not a numpy matrix")
            return

        X = np.append(np.ones(1), X)
        z = np.dot(X, self.W)
        return z

    def calculate_value(self, X):
        if not isinstance(X, np.ndarray):  # not necessary and can remove
            print("its not a numpy matrix")
            return

        X = np.append(np.ones(1), X)
        z = np.dot(X, self.W)
        result = self.activation_functions(z)

        if result > 0:
            return 1
        else:
            return 0

    def precision_calculate(self, X, y):
        ##
        # X matrix
        # y vector
        ##

        m = X.shape[0]
        corrects = 0
        for i in range(m):
            if self.calculate_value(X[i]) == y[i]:
                corrects += 1

        return corrects / m

    def plot_boundary_and_data(self, blues, reds, precision):
        ##
        # blues are pandas array
        # reds are pandas array
        ##
        x_axis = np.arange(-3, 9, 0.01)
        w = self.W
        y_axis = -w[0] / w[2] - x_axis * w[1] / w[2]

        plt.title(f'Adaline \n precision : {precision}  initial_learning_rate : {self.initial_learning_rate}')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.scatter(x_axis, y_axis)
        plt.scatter(blues['x'], blues['y'])
        plt.scatter(reds['x'], reds['y'])
        plt.show()

    def plot_curve_boundary_and_data(self, blues, reds, precision):
        ##
        # blues are pandas array
        # reds are pandas array
        ##
        x_axis = np.arange(-3, 9, 0.01)
        w = self.W

        x = np.linspace(-3, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        F = w[3] * X ** 2 + w[5] * X * Y + w[4] * Y ** 2 + w[1] * X + w[2] * Y + w[0]

        plt.title(f'Adaline \n  precision : {precision} initial_learning_rate : {self.initial_learning_rate}')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.contour(X, Y, F, levels=[0])  # take level set corresponding to 0
        plt.scatter(blues['x'], blues['y'])
        plt.scatter(reds['x'], reds['y'])
        plt.show()

    def plot_accuracy_history(self, precision):
        plt.title(f'Adaline \n train_precision : {self.history_err_train[-1]} test_precision : {self.history_err_test[-1]} \ninitial_learning_rate : {self.initial_learning_rate}')
        plt.xlabel("epoch")
        plt.ylabel("precision")
        x = np.arange(0, len(self.history_err_test), 1)
        plt.plot(x, self.history_err_train, label="train", alpha=0.5)
        plt.plot(x, self.history_err_test, label="test", alpha=0.5)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
