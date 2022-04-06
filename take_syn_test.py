import random
from collections import Counter, defaultdict
import numpy as np
from numpy import log
import scipy.linalg as scipy_linalg
import pandas as pd
from random import uniform


class TestTaker:

    def __init__(self):
        self.composes = None
        self.word2vec = None
        self.ec_scores = None
        self.cc_scores = None
        self.ew_scores = None
        self.ec_scores = None

    def get_compose_vecs(self):
        self.composes = pd.read_csv("EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt", delim_whitespace=True, quoting=3,
                                    header=None, index_col=0)
        # print(self.composes[:10])
        # print(self.composes.loc["auberge"])

    def get_word2vec_vecs(self):
        self.word2vec = pd.read_csv('GoogleNews-vectors-rcv_vocab.txt', delim_whitespace=True, quoting=3,
                                    header=None, index_col=0)
        # print(self.word2vec[-5:])
        # print(self.word2vec[:5])

    def euclidian_composes(self):
        results = defaultdict(dict)
        with open("multiple_choice_test.txt") as file:
            for i in range(1000):
                # print("hi_c")
                smallest_dist = 100000000000
                # choices = {}
                # print(file.readline())
                og_word = file.readline().split("_")[-1].rstrip("\n")
                # print(og_word)
                gold = file.readline().split("_")[-1].rstrip("\n")
                closest_word = gold
                # print(gold)
                results[og_word]["correct"] = gold
                if og_word in self.composes.index:
                    if gold in self.composes.index:
                        # print(og_word)
                        dist = scipy_linalg.norm(self.composes.loc[og_word] - self.composes.loc[gold])
                        if dist < smallest_dist:
                            smallest_dist = dist
                    else:
                        dist = scipy_linalg.norm(self.composes.loc[og_word] - np.zeros(500))
                        if dist < smallest_dist:
                            smallest_dist = dist
                else:
                    if gold in self.composes.index:
                        # print(og_word)
                        dist = scipy_linalg.norm(np.zeros(500) - self.composes.loc[gold])
                        if dist < smallest_dist:
                            smallest_dist = dist
                    else:
                        dist = 1
                        if dist < smallest_dist:
                            smallest_dist = dist
                if og_word in self.composes.index:
                    for j in range(3):
                        word = file.readline().split("_")[-1].rstrip("\n")
                        if word in self.composes.index:
                            dist = scipy_linalg.norm(self.composes.loc[og_word] -
                                                     self.composes.loc[word])
                            if dist < smallest_dist:
                                smallest_dist = dist
                                closest_word = word
                        else:
                            dist = scipy_linalg.norm(self.composes.loc[og_word] - np.zeros(500))
                            if dist < smallest_dist:
                                smallest_dist = dist
                                closest_word = word
                else:
                    for i in range(3):
                        word = file.readline().split("_")[-1].rstrip("\n")
                        if word in self.composes.index:
                            dist = scipy_linalg.norm(np.zeros(500) -
                                                     self.composes.loc[word])
                            if dist < smallest_dist:
                                smallest_dist = dist
                                closest_word = word
                        else:
                            dist = 1
                            if dist < smallest_dist:
                                smallest_dist = dist
                                closest_word = word

                # file.readline()
                results[og_word]["predicted"] = closest_word
        print(results)
        right = 0
        total = 0
        for q in results:
            if results[q]['correct'] == results[q]['predicted']:
                right += 1
                total += 1
            else:
                total += 1

        result = right / total

        print("Accuracy for Euclidian COMPOSES: " + str(result))

    def cosine_composes(self):
        results = defaultdict(dict)
        with open("multiple_choice_test.txt") as file:
            for i in range(1000):

                closest_score = -1

                og_word = file.readline().split("_")[-1].rstrip("\n")

                gold = file.readline().split("_")[-1].rstrip("\n")
                closest_word = gold

                results[og_word]["correct"] = gold
                # print(og_word)
                if og_word in self.composes.index:
                    if gold in self.composes.index:
                        # print(og_word)
                        angle = (self.composes.loc[og_word].dot(self.composes.loc[gold]) /
                                 (scipy_linalg.norm(self.composes.loc[og_word]) *
                                  scipy_linalg.norm(self.composes.loc[gold])))
                        if angle > closest_score:
                            closest_score = angle
                            closest_word = gold
                    else:
                        angle = 0
                        if angle > closest_score:
                            closest_score = angle
                            closest_word = gold
                else:
                    angle = 0
                    if angle > closest_score:
                        closest_score = angle
                        closest_word = gold

                if og_word in self.composes.index:
                    for p in range(3):
                        word = file.readline().split("_")[-1].rstrip("\n")
                        if word in self.composes.index:
                            angle = scipy_linalg.norm(self.composes.loc[og_word].dot(self.composes.loc[word]) /
                                                      (scipy_linalg.norm(self.composes.loc[og_word]) *
                                                       scipy_linalg.norm(self.composes.loc[word])))

                            if angle > closest_score:
                                closest_score = angle
                                closest_word = word
                        else:
                            angle = 0
                            if angle > closest_score:
                                closest_score = angle
                                closest_word = word
                else:
                    for c in range(3):
                        word = file.readline().split("_")[-1].rstrip("\n")
                        angle = 0
                        if angle > closest_score:
                            closest_score = angle
                            closest_word = word

                # file.readline()
                results[og_word]["predicted"] = closest_word
        print(results)
        right = 0
        total = 0
        for q in results:
            if results[q]['correct'] == results[q]['predicted']:
                right += 1
                total += 1
            else:
                total += 1

        result = right / total

        print("Accuracy for Cosine COMPOSES: " + str(result))

    def euclidian_word2vec(self):
        results = defaultdict(dict)
        with open("multiple_choice_test.txt") as file:
            for i in range(1000):
                smallest_dist = 1000000
                og_word = file.readline().split("_")[-1].rstrip("\n")

                gold = file.readline().split("_")[-1].rstrip("\n")
                closest_word = gold

                results[og_word]["correct"] = gold
                if og_word in self.word2vec.index:
                    if gold in self.word2vec.index:
                        # print(og_word)
                        dist = scipy_linalg.norm(self.word2vec.loc[og_word] - self.word2vec.loc[gold])
                        if dist < smallest_dist:
                            smallest_dist = dist
                    else:
                        dist = scipy_linalg.norm(self.word2vec.loc[og_word] - np.zeros(300))
                        if dist < smallest_dist:
                            smallest_dist = dist
                else:
                    if gold in self.word2vec.index:
                        # print(og_word)
                        dist = scipy_linalg.norm(np.zeros(300) - self.word2vec.loc[gold])
                        if dist < smallest_dist:
                            smallest_dist = dist
                    else:
                        dist = 1
                        if dist < smallest_dist:
                            smallest_dist = dist
                if og_word in self.word2vec.index:
                    for i in range(3):
                        word = file.readline().split("_")[-1].rstrip("\n")
                        if word in self.word2vec.index:
                            dist = scipy_linalg.norm(self.word2vec.loc[og_word] -
                                                     self.word2vec.loc[word])
                            if dist < smallest_dist:
                                smallest_dist = dist
                                closest_word = word
                        else:
                            dist = scipy_linalg.norm(self.word2vec.loc[og_word] -
                                                     np.zeros(300))
                            if dist < smallest_dist:
                                smallest_dist = dist
                                closest_word = word
                else:
                    for i in range(3):
                        word = file.readline().split("_")[-1].rstrip("\n")
                        if word in self.word2vec.index:
                            dist = scipy_linalg.norm(np.zeros(300) -
                                                     self.word2vec.loc[word])
                            if dist < smallest_dist:
                                smallest_dist = dist
                                closest_word = word
                        else:
                            dist = 1
                            if dist < smallest_dist:
                                smallest_dist = dist
                                closest_word = word

                # file.readline()
                results[og_word]["predicted"] = closest_word
        print(results)
        right = 0
        total = 0
        for q in results:
            if results[q]['correct'] == results[q]['predicted']:
                right += 1
                total += 1
            else:
                total += 1

        result = right / total

        print("Accuracy for Euclidian Word2Vec: " + str(result))

    def cosine_word2vec(self):
        results = defaultdict(dict)
        with open("multiple_choice_test.txt") as file:
            for i in range(1000):

                closest_score = -1

                og_word = file.readline().split("_")[-1].rstrip("\n")

                gold = file.readline().split("_")[-1].rstrip("\n")
                closest_word = gold

                results[og_word]["correct"] = gold
                # print(og_word)
                if og_word in self.word2vec.index:
                    if gold in self.word2vec.index:
                        # print(og_word)
                        angle = (self.word2vec.loc[og_word].dot(self.word2vec.loc[gold]) /
                                 (scipy_linalg.norm(self.word2vec.loc[og_word]) *
                                  scipy_linalg.norm(self.word2vec.loc[gold])))
                        if angle > closest_score:
                            closest_score = angle
                            closest_word = gold
                    else:
                        angle = 0
                        if angle > closest_score:
                            closest_score = angle
                            closest_word = gold
                else:
                    angle = 0
                    if angle > closest_score:
                        closest_score = angle
                        closest_word = gold

                if og_word in self.word2vec.index:
                    for p in range(3):
                        word = file.readline().split("_")[-1].rstrip("\n")
                        if word in self.word2vec.index:
                            angle = scipy_linalg.norm(self.word2vec.loc[og_word].dot(self.word2vec.loc[word]) /
                                                      (scipy_linalg.norm(self.word2vec.loc[og_word]) *
                                                       scipy_linalg.norm(self.word2vec.loc[word])))

                            if angle > closest_score:
                                closest_score = angle
                                closest_word = word
                        else:
                            angle = 0
                            if angle > closest_score:
                                closest_score = angle
                                closest_word = word
                else:
                    for c in range(3):
                        word = file.readline().split("_")[-1].rstrip("\n")
                        angle = 0
                        if angle > closest_score:
                            closest_score = angle
                            closest_word = word

                # file.readline()
                results[og_word]["predicted"] = closest_word
        print(results)
        right = 0
        total = 0
        for q in results:
            if results[q]['correct'] == results[q]['predicted']:
                right += 1
                total += 1
            else:
                total += 1

        result = right / total

        print("Accuracy for Cosine Word2Vec: " + str(result))

    def analogy_euclidian_composes(self):
        correct_answers = 0
        answer_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

        with open("SAT_package_V3.txt") as file:
            for k in range(41):
                file.readline()
            for i in range(374):
                x = 0
                idx = 0
                answer = ''
                answer_score = float(1000000)
                file.readline()
                file.readline()
                line = file.readline().split()
                # print(line)
                stem_pair = line[:2]
                # print(stem_pair)
                if stem_pair[0] in self.composes.index and stem_pair[1] in self.composes.index:
                    stem_dist = scipy_linalg.norm(self.composes.loc[stem_pair[0]] - self.composes.loc[stem_pair[1]])
                    x = 1
                elif stem_pair[0] in self.composes.index and stem_pair[1] not in self.composes.index:
                    stem_dist = scipy_linalg.norm(self.composes.loc[stem_pair[0]] - np.zeros(500))
                    x = 1
                elif stem_pair[0] not in self.composes.index and stem_pair[1] in self.composes.index:
                    stem_dist = scipy_linalg.norm(np.zeros(500) - self.composes.loc[stem_pair[1]])
                    x = 1
                else:
                    stem_dist = 10
                    for h in range(5):
                        # print("hello")
                        file.readline().split()
                        choice_pair = line[:2]
                        if choice_pair[0] in self.composes.index and choice_pair[1] in self.composes.index:
                            choice_dist = scipy_linalg.norm(self.composes.loc[choice_pair[0]] -
                                                            self.composes.loc[choice_pair[1]])
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        elif choice_pair[0] in self.composes.index and choice_pair[1] not in self.composes.index:
                            choice_dist = scipy_linalg.norm(self.composes.loc[choice_pair[0]] -
                                                            np.zeros(500))
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        elif choice_pair[0] not in self.composes.index and choice_pair[1] in self.composes.index:
                            choice_dist = scipy_linalg.norm(np.zeros(500) -
                                                            self.composes.loc[choice_pair[1]])
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        else:
                            choice_dist = 10
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        idx += 1
                    real_answer = file.readline()
                    if real_answer.strip() == answer:
                        correct_answers += 1

                if x == 1:
                    for j in range(5):
                        line = file.readline().split()
                        # print(line)
                        choice_pair = line[:2]
                        if choice_pair[0] in self.composes.index and choice_pair[1] in self.composes.index:
                            choice_dist = scipy_linalg.norm(self.composes.loc[choice_pair[0]] -
                                                            self.composes.loc[choice_pair[1]])
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        elif choice_pair[0] in self.composes.index and choice_pair[1] not in self.composes.index:
                            choice_dist = scipy_linalg.norm(self.composes.loc[choice_pair[0]] -
                                                            np.zeros(500))
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        elif choice_pair[0] not in self.composes.index and choice_pair[1] in self.composes.index:
                            choice_dist = scipy_linalg.norm(np.zeros(500) -
                                                            self.composes.loc[choice_pair[1]])
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        else:
                            choice_dist = 10
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        idx += 1
                    real_answer = file.readline()
                    # print(real_answer)
                    # print(answer)
                    if real_answer.strip() == answer:
                        correct_answers += 1
        print("Accuracy for COMPOSES Euclidian Analogies: " + str(correct_answers) + "/374")

    def analogy_cosine_composes(self):
        correct_answers = 0
        answer_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

        with open("SAT_package_V3.txt") as file:
            for k in range(41):
                file.readline()
            for i in range(374):
                x = 0
                idx = 0
                answer = ''
                answer_score = float(1000000)
                file.readline()
                file.readline()
                line = file.readline().split()
                # print(line)
                stem_pair = line[:2]
                if stem_pair[0] in self.composes.index and stem_pair[1] in self.composes.index:
                    stem_angle = scipy_linalg.norm(
                        self.composes.loc[stem_pair[0]].dot(self.composes.loc[stem_pair[1]]) /
                        (scipy_linalg.norm(self.composes.loc[stem_pair[0]]) *
                         scipy_linalg.norm(self.composes.loc[stem_pair[1]])))
                    x = 1
                else:
                    stem_angle = random.uniform(-1, 1)
                    for h in range(5):
                        # print("hello")
                        file.readline().split()
                        choice_pair = line[:2]
                        if choice_pair[0] in self.composes.index and choice_pair[1] in self.composes.index:
                            choice_angle = scipy_linalg.norm(
                                self.composes.loc[choice_pair[0]].dot(self.composes.loc[choice_pair[1]]) /
                                (scipy_linalg.norm(self.composes.loc[choice_pair[0]]) *
                                 scipy_linalg.norm(self.composes.loc[choice_pair[1]])))
                            dist = stem_angle - choice_angle
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        else:
                            dist = stem_angle
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        idx += 1
                    real_answer = file.readline()
                    if real_answer.strip() == answer:
                        correct_answers += 1

                if x == 1:
                    for j in range(5):
                        line = file.readline().split()
                        # print(line)
                        choice_pair = line[:2]
                        if choice_pair[0] in self.composes.index and choice_pair[1] in self.composes.index:
                            choice_angle = scipy_linalg.norm(
                                self.composes.loc[choice_pair[0]].dot(self.composes.loc[choice_pair[1]]) /
                                (scipy_linalg.norm(self.composes.loc[choice_pair[0]]) *
                                 scipy_linalg.norm(self.composes.loc[choice_pair[1]])))
                            dist = stem_angle - choice_angle
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        else:
                            dist = stem_angle
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        idx += 1
                    real_answer = file.readline()
                    # print(real_answer)
                    # print(answer)
                    if real_answer.strip() == answer:
                        correct_answers += 1
        print("Accuracy for COMPOSES cosine Analogies: " + str(correct_answers) + "/374")

    def analogy_euclidian_word2vec(self):
        correct_answers = 0
        answer_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

        with open("SAT_package_V3.txt") as file:
            for k in range(41):
                file.readline()
            for i in range(374):
                x = 0
                idx = 0
                answer = ''
                answer_score = float(1000000)
                file.readline()
                file.readline()
                line = file.readline().split()
                # print(line)
                stem_pair = line[:2]
                # print(stem_pair)
                if stem_pair[0] in self.word2vec.index and stem_pair[1] in self.word2vec.index:
                    stem_dist = scipy_linalg.norm(self.word2vec.loc[stem_pair[0]] - self.word2vec.loc[stem_pair[1]])
                    x = 1
                elif stem_pair[0] in self.word2vec.index and stem_pair[1] not in self.word2vec.index:
                    stem_dist = scipy_linalg.norm(self.word2vec.loc[stem_pair[0]] - np.zeros(300))
                    x = 1
                elif stem_pair[0] not in self.word2vec.index and stem_pair[1] in self.word2vec.index:
                    stem_dist = scipy_linalg.norm(np.zeros(300) - self.word2vec.loc[stem_pair[1]])
                    x = 1
                else:
                    stem_dist = 10
                    for h in range(5):
                        # print("hello")
                        file.readline().split()
                        choice_pair = line[:2]
                        if choice_pair[0] in self.word2vec.index and choice_pair[1] in self.word2vec.index:
                            choice_dist = scipy_linalg.norm(self.word2vec.loc[choice_pair[0]] -
                                                            self.word2vec.loc[choice_pair[1]])
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        elif choice_pair[0] in self.word2vec.index and choice_pair[1] not in self.word2vec.index:
                            choice_dist = scipy_linalg.norm(self.word2vec.loc[choice_pair[0]] -
                                                            np.zeros(300))
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        elif choice_pair[0] not in self.word2vec.index and choice_pair[1] in self.word2vec.index:
                            choice_dist = scipy_linalg.norm(np.zeros(300) -
                                                            self.word2vec.loc[choice_pair[1]])
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        else:
                            choice_dist = 10
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        idx += 1
                    real_answer = file.readline()
                    if real_answer.strip() == answer:
                        correct_answers += 1

                if x == 1:
                    for j in range(5):
                        line = file.readline().split()
                        # print(line)
                        choice_pair = line[:2]
                        if choice_pair[0] in self.word2vec.index and choice_pair[1] in self.word2vec.index:
                            choice_dist = scipy_linalg.norm(self.word2vec.loc[choice_pair[0]] -
                                                            self.word2vec.loc[choice_pair[1]])
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        elif choice_pair[0] in self.word2vec.index and choice_pair[1] not in self.word2vec.index:
                            choice_dist = scipy_linalg.norm(self.word2vec.loc[choice_pair[0]] -
                                                            np.zeros(300))
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        elif choice_pair[0] not in self.word2vec.index and choice_pair[1] in self.word2vec.index:
                            choice_dist = scipy_linalg.norm(np.zeros(300) -
                                                            self.word2vec.loc[choice_pair[1]])
                            dist = stem_dist - choice_dist
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        idx += 1
                    real_answer = file.readline()
                    # print(real_answer)
                    # print(answer)
                    if real_answer.strip() == answer:
                        correct_answers += 1
        print("Accuracy for Word2Vec Euclidian Analogies: " + str(correct_answers) + "/374")

    def analogy_cosine_word2vec(self):
        correct_answers = 0
        answer_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

        with open("SAT_package_V3.txt") as file:
            for k in range(41):
                file.readline()
            for i in range(374):
                x = 0
                idx = 0
                answer = ''
                answer_score = float(1000000)
                file.readline()
                file.readline()
                line = file.readline().split()
                # print(line)
                stem_pair = line[:2]
                if stem_pair[0] in self.word2vec.index and stem_pair[1] in self.word2vec.index:
                    stem_angle = scipy_linalg.norm(
                        self.word2vec.loc[stem_pair[0]].dot(self.word2vec.loc[stem_pair[1]]) /
                        (scipy_linalg.norm(self.word2vec.loc[stem_pair[0]]) *
                         scipy_linalg.norm(self.word2vec.loc[stem_pair[1]])))
                    x = 1
                else:
                    stem_angle = random.uniform(-1, 1)
                    for h in range(5):
                        # print("hello")
                        file.readline().split()
                        choice_pair = line[:2]
                        if choice_pair[0] in self.word2vec.index and choice_pair[1] in self.word2vec.index:
                            scipy_linalg.norm(
                                self.word2vec.loc[choice_pair[0]].dot(self.word2vec.loc[choice_pair[1]]) /
                                (scipy_linalg.norm(self.word2vec.loc[choice_pair[0]]) *
                                 scipy_linalg.norm(self.word2vec.loc[choice_pair[1]])))
                            dist = stem_angle - choice_angle
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        else:
                            dist = stem_angle
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        idx += 1
                    real_answer = file.readline()
                    if real_answer.strip() == answer:
                        correct_answers += 1

                if x == 1:
                    for j in range(5):
                        line = file.readline().split()
                        # print(line)
                        choice_pair = line[:2]
                        if choice_pair[0] in self.word2vec.index and choice_pair[1] in self.word2vec.index:
                            choice_angle = scipy_linalg.norm(
                                self.word2vec.loc[choice_pair[0]].dot(self.word2vec.loc[choice_pair[1]]) /
                                (scipy_linalg.norm(self.word2vec.loc[choice_pair[0]]) *
                                 scipy_linalg.norm(self.word2vec.loc[choice_pair[1]])))
                            dist = stem_angle - choice_angle
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        else:
                            dist = stem_angle
                            if abs(dist) < answer_score:
                                answer = answer_dict[idx]
                                answer_score = abs(dist)
                        idx += 1
                    real_answer = file.readline()
                    # print(real_answer)
                    # print(answer)
                    if real_answer.strip() == answer:
                        correct_answers += 1
        print("Accuracy for Word2Vec cosine Analogies: " + str(correct_answers) + "/374")


if __name__ == '__main__':
    test_taker = TestTaker()
    test_taker.get_compose_vecs()
    test_taker.get_word2vec_vecs()
    # test_taker.euclidian_composes()
    # test_taker.euclidian_word2vec()
    # test_taker.cosine_composes()
    # test_taker.cosine_word2vec()
    test_taker.analogy_euclidian_composes()
    test_taker.analogy_euclidian_word2vec()
    test_taker.analogy_cosine_composes()
    test_taker.analogy_cosine_word2vec()
