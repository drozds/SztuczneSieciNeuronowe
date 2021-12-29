from random import random
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, n_input, n_hidden, n_output, act_func, derivative):
        self.hidden_layer = [{} for i in range(n_hidden)]
        self.output_layer = [{} for i in range(n_output)]
        self.act_func = act_func
        self.derivative = derivative
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

    def initialize(self):
        for neuron in self.hidden_layer:
            neuron['weights'] = [random() for i in range(self.n_input)]
            bias = random()
            neuron['weights'].append(bias)
        for neuron in self.output_layer:
            neuron['weights'] = [
                random() for i in range(len(self.hidden_layer))
            ]
            bias = random()
            neuron['weights'].append(bias)

    def clear_all(self):
        for neuron in self.hidden_layer:
            neuron.clear()
        for neuron in self.output_layer:
            neuron.clear()

    def get_weighted_sum(self, weights, inputs):
        sum = weights[-1]
        for i in range(len(weights) - 1):
            sum += weights[i] * inputs[i]
        return sum

    def forward_propagate(self, data_row):
        hidden_layer_inputs = data_row
        output_layer_inputs = []
        outputs = []
        for neuron in self.hidden_layer:
            weighted_sum = self.get_weighted_sum(neuron['weights'],
                                                 hidden_layer_inputs)
            neuron['output'] = self.act_func(weighted_sum)
            output_layer_inputs.append(neuron['output'])
        for neuron in self.output_layer:
            weighted_sum = self.get_weighted_sum(neuron['weights'],
                                                 output_layer_inputs)
            neuron['output'] = self.act_func(weighted_sum)
            outputs.append(neuron['output'])
        return outputs

    def backward_propagate_error(self, expected):
        for j in range(len(self.output_layer)):
            neuron = self.output_layer[j]
            error_output = neuron['output'] - expected[j]
            neuron['delta'] = error_output * self.derivative(neuron['output'])
        for j in range(len(self.hidden_layer)):
            error_hidden = 0.0
            for neuron in self.output_layer:
                error_hidden += (neuron['weights'][j] * neuron['delta'])
            neuron = self.hidden_layer[j]
            neuron['delta'] = error_hidden * self.derivative(neuron['output'])
        return error_output, error_hidden

    def update_weights(self, data_row, l_rate):
        hidden_layer_inputs = data_row[:-1]
        for neuron in self.hidden_layer:
            for j in range(len(hidden_layer_inputs)):
                neuron['weights'][
                    j] -= l_rate * neuron['delta'] * hidden_layer_inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']
        output_layer_inputs = [
            neuron['output'] for neuron in self.hidden_layer
        ]
        for neuron in self.output_layer:
            for j in range(len(output_layer_inputs)):
                neuron['weights'][
                    j] -= l_rate * neuron['delta'] * output_layer_inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']

    def train(self, train_set, l_rate, n_epoch, n_outputs):
        sum_errors = []
        epochs = [i for i in range(1, n_epoch + 1)]
        for epoch in range(n_epoch):
            sum_error = 0
            for data_row in train_set:
                outputs = self.forward_propagate(data_row)
                expected = [0 for i in range(n_outputs)]
                expected[data_row[-1]] = 1
                sum_error += sum([(expected[i] - outputs[i])**2
                                  for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(data_row, l_rate)
            sum_errors.append(sum_error / len(train_set))

        # plt.plot(epochs, sum_errors)
        # plt.ylabel("MSE")
        # plt.xlabel("epoch")
        # plt.show()
