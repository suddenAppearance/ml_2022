import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class KNN:
    def __init__(self, dataset, n_neighbours=3):
        self.dataset = dataset
        self.n_neighbours = n_neighbours

    @staticmethod
    def dist(list1, list2):
        return np.sqrt(sum([(i - j) ** 2 for i, j in zip(list1, list2)]))

    @staticmethod
    def score(y_test, predictions):
        return accuracy_score(predictions, y_test)

    def predict(self, list2d):
        predictions = []
        for i in range(len(list2d)):
            distances = []
            targets = {}
            for j in range(len(self.dataset)):
                distances.append([self.dist(list2d[i], self.dataset[j][0]), j])
            distances = sorted(distances)

            for j in range(self.n_neighbours):
                index = distances[j][1]
                if targets.get(self.dataset[index][1]) is not None:
                    targets[self.dataset[index][1]] += 1
                else:
                    targets[self.dataset[index][1]] = 1

            predictions.append(max(targets, key=targets.get))

        return predictions


digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20)

dataset = [[x_train[i], y_train[i]] for i in range(len(x_train))]
knn = KNN(dataset)
pred = knn.predict(x_test)
print(knn.score(y_test, predictions=pred))
