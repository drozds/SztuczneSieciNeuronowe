from csv import reader


class Dataset:

    def __init__(self, filename, train_percent=0.6, cross=False, n_folds=0):
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
        columns_range = [{
            'min': min(column),
            'max': max(column)
        } for column in zip(*self.data)]
        for row in self.data:
            for i in range(len(row) - 1):
                row[i] = (row[i] - columns_range[i]['min']) / (
                    columns_range[i]['max'] - columns_range[i]['min'])

    def cross_validation_split(self):
        fold_size = int(len(self.data) / self.n_folds)
        for i in range(self.n_folds):
            fold = self.data[i * fold_size:(i + 1) * fold_size]
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
        class_values = list(set([row[-1] for row in self.data]))
        class_values.sort()
        for i, value in enumerate(class_values):
            self.classnames_map[value] = i
        for row in self.data:
            row[-1] = self.classnames_map[row[-1]]
        self.normalize()
        if self.cross:
            self.cross_validation_split()
        else:
            self.train_test_split()
