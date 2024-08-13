import numpy as np
from sklearn.neural_network import BernoulliRBM

from time import time

# Take a look at https://github.com/akoreman/Restricted-Boltzmann-Machine-Generative-Ising-Model


def magnetisation(grid):
    return np.sum(grid)/len(grid)


'''
Define a RBM, train it and use it to generate new lattices. Use these lattices to calculate the observables.
This process is done at each temperature in the range and at each number of hidden nodes that we want to calculate.
'''


def main():

    # TODO: define the number of hidden nodes, the number of tests
    hidden_nodes = 100
    n_tests = 100

    # TODO: download the dataset and evaluate the magnetisation (take a single configuration)
    training_vectors = np.load("magnetization_generation.npz")["vector"]
    # TODO: Instantiate the model and train it
    rbm = BernoulliRBM(n_components = hidden_nodes, learning_rate=0.01, verbose=0, n_iter=n_tests, batch_size=16)
    rbm.fit(training_vectors)  # function for training data

    # TODO: define a set of vectors randomly initialized, thus perform a Gibbs sampling
    # remember to define a set of steps which perform the Gibbs sampling through
    # remember to recall the n_test variable to set the number of random vectors
    number_of_vectors = 100
    size_of_vectors = training_vectors.shape[1]
    vector_test = np.random.randint(2, size=(number_of_vectors, size_of_vectors))
    for _ in range(1000):
        equilibration_steps = rbm.gibbs(vector_test)

    # TODO: evaluate mean
    m = np.mean(equilibration_steps,axis=1)
    # TODO: evaluate standard deviation
    sd = np.std(equilibration_steps,axis=1)

    print("mean is ", m)
    print("std is ", sd)
if __name__ == "__main__":
    main()
