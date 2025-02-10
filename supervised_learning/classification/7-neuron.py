import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    Class that defines a single neuron for binary classification
    """
    def __init__(self, nx):
        """
        Initializes the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

    def forward_prop(self, X):
        """
        Performs forward propagation using a sigmoid activation function
        """
        self.A = 1 / (1 + np.exp(-(np.matmul(self.W, X) + self.b)))
        return self.A

    def cost(self, Y, A):
        """
        Computes the logistic regression cost function
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8)) / m

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one pass of gradient descent
        """
        m = Y.shape[1]
        dW = np.matmul((A - Y), X.T) / m
        db = np.sum(A - Y) / m
        self.W -= alpha * dW
        self.b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")
        
        costs = []
        iterations_list = []

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)
                iterations_list.append(i)
        
        # Capture final cost after training
        final_cost = self.cost(Y, self.A)
        costs.append(final_cost)
        iterations_list.append(iterations)
        if verbose:
            print(f"Cost after {iterations} iterations: {final_cost}")

        if graph:
            plt.plot(iterations_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        
        return self.evaluate(X, Y)
