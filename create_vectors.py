from collections import Counter
import numpy as np
from numpy import log
import scipy.linalg as scipy_linalg


class DistributionalVectorCreator:

    def __init__(self):
        self.co_occurrence = None
        self.scalar = 10
        self.ppmi_matrix = None
        self.feature_dict = {}
        self.total_words = 0
        self.word_counts = Counter()

    def create_matrices(self, filename):
        feat_count = 0
        with open(filename, encoding="UTF-8") as f:
            for line in f:
                self.total_words += len(line)
                for token in line.strip("\n").split(" "):
                    self.word_counts[token] += 1
                    if token not in self.feature_dict:
                        self.feature_dict[token] = feat_count
                        feat_count += 1
            self.feature_dict = {'bite': 0, 'dogs': 1, 'feed': 2, 'like': 3, 'men': 4, 'the': 5, 'women': 6}

            # set dimensions equal to V by V
            self.co_occurrence = np.zeros((feat_count, feat_count))

            with open(filename, encoding="UTF-8") as f:
                for line in f:
                    line = line.strip("\n").split(" ")

                    for a, b in zip(line, line[1:]):
                        self.co_occurrence[self.feature_dict[a], self.feature_dict[b]] += 1
                        if a != b:
                            self.co_occurrence[self.feature_dict[b], self.feature_dict[a]] += 1

        # multiply each value by 10 and smooth by adding 1 to each element
        self.co_occurrence *= self.scalar
        self.co_occurrence += 1

        print(self.co_occurrence)

        # convert to positive pointwise mutual information matrix
        self.ppmi_matrix = np.zeros((feat_count, feat_count))

        c = 0
        for word_row in self.co_occurrence:
            r = 0
            for context_count in word_row:
                if r < feat_count:
                    ppmi = max(log((context_count / np.sum(self.co_occurrence)) /
                                   ((np.sum(self.co_occurrence[r, :]) / np.sum(self.co_occurrence)) *
                                    (np.sum(self.co_occurrence[:, c]) / np.sum(self.co_occurrence)))), 0)
                    self.ppmi_matrix[c, r] = ppmi
                    r += 1
            c += 1

        print(self.ppmi_matrix)

        # Good collocation pairs have high PMI because the probability of co-occurrence is only slightly lower than
        # the probabilities of occurrence of each word. Conversely, a pair of words whose probabilities
        # of occurrence are considerably higher than their probability of co-occurrence gets a small PMI score.

    def euclidian_distance(self):
        distances = [scipy_linalg.norm(self.ppmi_matrix[self.feature_dict["women"]] -
                                       self.ppmi_matrix[self.feature_dict["men"]]),
                     scipy_linalg.norm(self.ppmi_matrix[self.feature_dict["women"]] -
                                       self.ppmi_matrix[self.feature_dict["dogs"]]),
                     scipy_linalg.norm(self.ppmi_matrix[self.feature_dict["men"]] -
                                       self.ppmi_matrix[self.feature_dict["dogs"]]),
                     scipy_linalg.norm(self.ppmi_matrix[self.feature_dict["feed"]] -
                                       self.ppmi_matrix[self.feature_dict["like"]]),
                     scipy_linalg.norm(self.ppmi_matrix[self.feature_dict["feed"]] -
                                       self.ppmi_matrix[self.feature_dict["bite"]]),
                     scipy_linalg.norm(self.ppmi_matrix[self.feature_dict["like"]] -
                                       self.ppmi_matrix[self.feature_dict["bite"]])]

        print(distances)

        U, E, Vt = scipy_linalg.svd(self.ppmi_matrix, full_matrices=False)
        E = np.diag(E)  # compute E
        print(np.allclose(self.ppmi_matrix, U.dot(E).dot(Vt)))

        V = Vt.T  # compute V = conjugate transpose of Vt
        reduced_PPMI = self.ppmi_matrix.dot(V[:, 0:3])

        reduced_distances = [scipy_linalg.norm(reduced_PPMI[self.feature_dict["women"]] -
                                               reduced_PPMI[self.feature_dict["men"]]),
                             scipy_linalg.norm(reduced_PPMI[self.feature_dict["women"]] -
                                               reduced_PPMI[self.feature_dict["dogs"]]),
                             scipy_linalg.norm(reduced_PPMI[self.feature_dict["men"]] -
                                               reduced_PPMI[self.feature_dict["dogs"]]),
                             scipy_linalg.norm(reduced_PPMI[self.feature_dict["feed"]] -
                                               reduced_PPMI[self.feature_dict["like"]]),
                             scipy_linalg.norm(reduced_PPMI[self.feature_dict["feed"]] -
                                               reduced_PPMI[self.feature_dict["bite"]]),
                             scipy_linalg.norm(reduced_PPMI[self.feature_dict["like"]] -
                                               reduced_PPMI[self.feature_dict["bite"]])]

        print(reduced_distances)


if __name__ == '__main__':
    vec = DistributionalVectorCreator()
    vec.create_matrices("dist_sim_data.txt")
    vec.euclidian_distance()
