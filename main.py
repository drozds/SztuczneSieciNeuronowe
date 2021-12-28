from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

class Dataset:
  def __init__(self, filename, train_percent = 0.6, cross = False, n_folds = 0):
      self.data = []
      self.classnames_map = {}
      self.filename = filename
      self.cross = cross
      if cross:
          self.n_folds = n_folds
          self.folds = []
      else:
          self.train_percent = train_percent
          self.train = []
          self.test = []

  def load(self):
      with open(self.filename, 'r') as file:
          csv_reader = reader(file, delimiter=';')
          for index, row in enumerate(csv_reader):
              if index == 0:
                  continue
              if not row:
                  continue
              self.data.append(row)

  def normalize(self):
      columns_range = [{ 'min': min(column), 'max': max(column) } for column in zip(*self.data)]
      for row in self.data:
          for i in range(len(row) - 1):
              row[i] = (row[i] - columns_range[i]['min']) / (columns_range[i]['max'] - columns_range[i]['min'])
  
  def cross_validation_split(self):
      fold_size = int(len(self.data) / self.n_folds)
      for i in range(self.n_folds):
          fold = self.data[i * fold_size : (i + 1) * fold_size]
          self.folds.append(fold)
  
  def train_test_split(self):
      data_length = len(self.data)
      self.train = self.data[:round(data_length * self.train_percent)]
      self.test = self.data[round(data_length * self.train_percent):]
  
  def organize(self):
      self.load()
      for i in range(len(self.data[0]) - 1):
          for row in self.data:
              row[i] = float(row[i].strip())
      class_values = set([row[-1] for row in self.data])
      for i, value in enumerate(class_values):
          self.classnames_map[value] = i
      for row in self.data:
          row[-1] = self.classnames_map[row[-1]]
      self.normalize()
      if self.cross:
          self.cross_validation_split()
      else:
          self.train_test_split()

      

sigmoid_act_func = lambda x: 1.0 / (1.0 + exp(-x))
sigmoid_derivative = lambda x: x * (1 - x)

class NeuralNetwork:
  def __init__(self, n_input, n_hidden, n_output, act_func, derivative):
      self.hidden_layer = [{} for i in range(n_hidden)]
      self.output_layer = [{} for i in range(n_output)]
      self.act_func = act_func
      self.derivative = derivative


  def initialize(self):
      for neuron in self.hidden_layer:
          neuron['weights'] = [random() for i in range(len(self.hidden_layer))]
          bias = random()
          neuron['weights'].append(bias)
      for neuron in self.output_layer:
          neuron['weights'] = [random() for i in range(len(self.output_layer))]
          bias = random()
          neuron['weights'].append(bias)

  def clear_all(self):
      for neuron in self.hidden_layer:
          neuron = {}
      for neuron in self.output_layer:
          neuron = {}
  

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
          weighted_sum = self.get_weighted_sum(neuron['weights'], hidden_layer_inputs)
          neuron['output'] = self.act_func(weighted_sum)
          output_layer_inputs.append(neuron['output'])
      for neuron in self.output_layer:
          weighted_sum = self.get_weighted_sum(neuron['weights'], output_layer_inputs)
          neuron['output'] = self.act_func(weighted_sum)
          outputs.append(neuron['output'])
      return outputs
  

  def backward_propagate_error(self, expected):
      for j in range(len(self.output_layer)):
          neuron = self.output_layer[j]
          error = neuron['output'] - expected[j]
          neuron['delta'] = error * self.derivative(neuron['output'])
      for j in range(len(self.hidden_layer)):
          error = 0.0
          for neuron in self.output_layer:
              error += (neuron['weights'][j] * neuron['delta'])
          neuron = self.hidden_layer[j]
          neuron['delta'] = error * self.derivative(neuron['output'])

  def update_weights(self, data_row, l_rate):
      hidden_layer_inputs = data_row[:-1]
      for neuron in self.hidden_layer:
          for j in range(len(hidden_layer_inputs)):
              neuron['weights'][j] -= l_rate * neuron['delta'] * hidden_layer_inputs[j]
          neuron['weights'][-1] -= l_rate * neuron['delta']
      output_layer_inputs = [neuron['output'] for neuron in self.hidden_layer]
      for neuron in self.output_layer:
          for j in range(len(output_layer_inputs)):
              neuron['weights'][j] -= l_rate * neuron['delta'] * output_layer_inputs[j]
          neuron['weights'][-1] -= l_rate * neuron['delta']

  def train(self, train_set, l_rate, n_epoch, n_outputs):
      for epoch in range(n_epoch):
          for data_row in train_set:
              outputs = self.forward_propagate(data_row)
              expected = [0 for i in range(n_outputs)]
              expected[data_row[-1]] = 1
              self.backward_propagate_error(expected)
              self.update_weights(data_row, l_rate)

def accuracy_metric(actual, guessed):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == guessed[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0  
   
def main():
    filename = 'winequality-white.csv'
    dataset = Dataset(filename, cross=True, n_folds=5)
    dataset.organize()
    n_inputs = len(dataset.data[0]) - 1
    n_outputs = len(dataset.classnames_map)

    network = NeuralNetwork(n_inputs, 8, n_outputs, sigmoid_act_func, sigmoid_derivative)

    scores = []

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
        network.train(train_set, l_rate=0.1, n_epoch=10, n_outputs=n_outputs)
        guesses = []
        for row in test_set:
            outputs = network.forward_propagate(row)
            guess = outputs.index(max(outputs))
            guesses.append(guess)

        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, guesses)
        scores.append(accuracy)
        network.clear_all()
    
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))


if __name__ == '__main__':
  main()
