# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from collections import Counter
from random import shuffle, choice
import numpy as np
import scipy
import operator


class MaxEnt(Classifier):
    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(instances, dev_instances, 0.0001, 30)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Minibatch Stochastic Gradient."""

        all_instances = train_instances + dev_instances
        shuffle(train_instances)
        minibatch_list = [train_instances[i:i + batch_size] for i in range(0, len(train_instances), batch_size)]

        temp_list = []
        feature_list = []
        for instance in all_instances:
            for word in instance.features():
                temp_list.append(word)
        freqs = Counter(temp_list)
        for (word, count) in freqs.most_common(1000):
            feature_list.append(word)
        print("Features:", feature_list)
        feature_list.append('BIAS')

        label_list = []
        for instance in all_instances:
            if instance.label not in label_list:
                label_list.append(instance.label)

        parameter_matrix = np.zeros((len(label_list), len(feature_list)))
        feature_dict = {}
        label_dict = {}
        for i, feature in enumerate(feature_list):
            feature_dict[feature] = i
        for j, label in enumerate(label_list):
            label_dict[label] = j
        self.model = (parameter_matrix, feature_dict, label_dict)

        for instance in all_instances:
            for feature in instance.features():
                if feature in feature_dict:
                    instance.pivot_list.append(feature_dict[feature])
            instance.pivot_list.append(feature_dict['BIAS'])

        converged = False
        total_iteration = 0
        tracker = 0
        best_matrix = parameter_matrix
        best_accuracy = 0

        while not converged:
            print("Iteration:", total_iteration, "Current Accuracy:", self.accuracy(dev_instances), "Log-Likelihood:",
                  self.log_likelihood(dev_instances))
            total_iteration += 1
            tracker += 1

            for minibatch in minibatch_list:
                gradient = self.compute_gradient(minibatch)
                parameter_matrix += gradient * learning_rate

            self.model = (parameter_matrix, feature_dict, label_dict)

            acc = self.accuracy(dev_instances)
            if acc > best_accuracy:
                best_accuracy = acc
                best_matrix = parameter_matrix
                tracker = 0

            if tracker >= 5:
                converged = True

        self.model = (best_matrix, feature_dict, label_dict)

    def compute_gradient(self, minibatch):
        return self.empirical_expectation(minibatch) - self.model_expectation(minibatch)

    def empirical_expectation(self, minibatch):
        parameter_matrix, feature_dict, label_dict = self.model
        expectation_matrix = np.zeros((len(label_dict), len(feature_dict)))
        for instance in minibatch:
            label_index = label_dict[instance.label]
            for feature_index in instance.pivot_list:
                expectation_matrix[label_index, feature_index] += 1
        return expectation_matrix

    def model_expectation(self, minibatch):
        parameter_matrix, feature_dict, label_dict = self.model
        expectation_matrix = np.zeros((len(label_dict), len(feature_dict)))
        for instance in minibatch:
            pd_dict = self.posterior_distribution(instance)
            for label in pd_dict:
                label_index = label_dict[label]
                for feature_index in instance.pivot_list:
                    expectation_matrix[label_index, feature_index] += pd_dict[label]
        return expectation_matrix

    def accuracy(self, instance_list):
        correct = 0
        for instance in instance_list:
            if self.classify(instance) == instance.label:
                correct += 1
        return correct / len(instance_list)

    def classify(self, instance):
        parameter_matrix, feature_dict, label_dict = self.model

        instance.pivot_list = []
        for feature in instance.features():
            if feature in feature_dict:
                instance.pivot_list.append(feature_dict[feature])
        instance.pivot_list.append(feature_dict['BIAS'])
        pd_dict = self.posterior_distribution(instance)

        if pd_dict:
            maximum = max(pd_dict.values())
            key_list = list(filter(lambda x: pd_dict[x] == maximum, pd_dict.keys()))
            if len(key_list) > 1:
                return choice(key_list)
            else:
                return key_list[0]
        else:
            return choice(label_dict.keys())

    def posterior_distribution(self, instance):
        parameter_matrix, feature_dict, label_dict = self.model

        pd_dict = {}
        total = []
        current = []

        for label in label_dict:
            for pivot in instance.pivot_list:
                total.append(parameter_matrix[label_dict[label], pivot])

        for label in label_dict:
            for pivot in instance.pivot_list:
                current.append(parameter_matrix[label_dict[label], pivot])
            pd_dict[label] = np.exp(sum(current) - scipy.misc.logsumexp(total))
        return pd_dict

    def log_likelihood(self, instance_list):
        likelihood = 0.0
        for instance in instance_list:
            pd_dict = self.posterior_distribution(instance)
            if pd_dict[instance.label] != 0:
                likelihood += np.log(pd_dict[instance.label])
        return likelihood
