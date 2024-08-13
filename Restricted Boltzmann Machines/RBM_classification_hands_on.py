from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from statistics import mode

import numpy as np

# Take a look at https://scikit-learn.org/stable/auto_examples/neural_networks/plot_rbm_logistic_classification.html


def magnetisation(grid):
    return np.sum(grid)/len(grid)


def test_model(rbm, v):

    v_tmp = v.copy()
    clamped = v[:, :-3]
    for _ in range(1000):
        # Compute the v_k
        v_tmp = rbm.gibbs(v_tmp)
        v_tmp[:, :-3] = clamped
    # Using three predicting elements, cause the mode is computed on them (the most frequent predicted value)
    return [mode(elem) for elem in v_tmp[:, -3:]]


def main():
    # TODO: download the datasets
    # 1000 configurations of 1000 spins each one
    vector2 = np.load("magnetization_02.npz")["arr_0"]
    vector8 = np.load("magnetization_08.npz")["arr_0"]
    print(vector8.shape)

    # TODO: attach the labels at the end of each spin vector (numpy concatenation)
    # Hint: np.c_ and np.r_ could turn to be useful
    np.r_(vector2, [0, 0, 0])
    
    training_data =
    test_size=0.2
    X_train, X_test, Y_train, Y_test = train_test_split(training_data, training_data[:,-1], test_size=test_size, random_state=0)
    # TODO: Instantiate the model and train it

    # TODO: Call the test_model function on x test
    Y_pred = ...

    # TODO: Compare the output with y_train by using balanced_accuracy_score method

if __name__ == "__main__":
    main()
