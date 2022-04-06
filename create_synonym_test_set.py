from random import randint
import pandas as pd

syn_df = pd.read_csv("EN_syn_verb.txt", sep='\t')

with open("multiple_choice_test.txt", "a") as test_file:
    for idx, row in syn_df.iterrows():
        test_file.write(row["Input.word"] + "\n")
        if row["Answer.suggestion"] != '0':
            test_file.write(row["Answer.suggestion"] + "\n")
        else:
            if syn_df["Input.word"][idx+1] == row["Input.word"]:
                test_file.write(syn_df["Answer.suggestion"][idx+1] + "\n")
            else:
                test_file.write(syn_df["Answer.suggestion"][idx - 1] + "\n")
        for i in range(4):
            x = 0
            while x == 0:
                if idx < 986:
                    choice = randint(idx, 996)
                else:
                    choice = randint(0, idx-10)
                if syn_df["Input.word"][choice] != 0:
                    test_file.write(syn_df["Input.word"][choice] + "\n")
                    x = 1
        # test_file.write("\n")

    for i in range(3):
        choice = randint(1, 996)
        test_file.write(syn_df["Input.word"][choice] + "\n")
        if syn_df["Answer.suggestion"][choice] != '0':
            test_file.write(syn_df["Answer.suggestion"][choice] + "\n")
        else:
            if syn_df["Input.word"][choice + 1] == syn_df["Input.word"][choice]:
                test_file.write(syn_df["Answer.suggestion"][choice + 1] + "\n")
            else:
                test_file.write(syn_df["Answer.suggestion"][choice - 1] + "\n")
        for k in range(4):
            x = 0
            while x == 0:
                if choice < 986:
                    choice = randint(choice, 996)
                else:
                    choice = randint(0, choice - 10)
                if syn_df["Answer.suggestion"][choice] != '0':
                    test_file.write(syn_df["Answer.suggestion"][choice] + "\n")
                    x = 1
        # test_file.write("\n")

test_file.close()
