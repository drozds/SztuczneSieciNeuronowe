from math import exp

from dataset import Dataset
from neuralnetwork import NeuralNetwork

sigmoid_act_func = lambda x: 1.0 / (1.0 + exp(-x))
sigmoid_derivative = lambda x: x * (1 - x)


def evaluate_results(actual, guessed):
    correct = 0
    close = {}

    for i in range(len(actual)):
        if actual[i] == guessed[i]:
            correct += 1
        else:
            closeness = abs(actual[i] - guessed[i])
            if not closeness in close:
                close[closeness] = 0
            close[closeness] += 1

    percent_correct = correct / float(len(actual)) * 100.0
    percents_close = {
        k: v / float(len(actual)) * 100.0
        for k, v in close.items()
    }
    return percent_correct, percents_close


def get_dataset(*args, **kwargs):
    dataset = Dataset(*args, **kwargs)
    dataset.organize()
    n_inputs = len(dataset.data[0]) - 1
    n_outputs = len(dataset.classnames_map)
    return dataset, n_inputs, n_outputs


def main():
    filename = 'winequality-white.csv'
    dataset, n_inputs, n_outputs = get_dataset(filename, cross=True, n_folds=5)

    network = NeuralNetwork(n_inputs, 4, n_outputs, sigmoid_act_func,
                            sigmoid_derivative)

    scores = []
    errors_of_one = []

    for fold in dataset.folds:
        train_set = list(dataset.folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        network.initialize()
        network.train(train_set, l_rate=0.5, n_epoch=10, n_outputs=n_outputs)
        guesses = []
        for row in test_set:
            outputs = network.forward_propagate(row)
            guess = outputs.index(max(outputs))
            guesses.append(guess)

        actual = [row[-1] for row in fold]
        accuracy, close_guesses = evaluate_results(actual, guesses)
        scores.append(accuracy)
        errors_of_one.append(close_guesses[1])
        network.clear_all()

    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
    print("close:", errors_of_one)
    print('Close guesses (error of 1 point): %.3f%%' %
          (sum(errors_of_one) / float(len(errors_of_one))))


if __name__ == '__main__':
    main()
    epochs = [5, 10, 20, 100]
    learn_rates = [0.1, 0.2, 0.5, 1, 2, 5]
    mses = [50.889, 51.64, 51.72, 52.95]
    closes = [42.41, 42, 41.98, 41.53]
