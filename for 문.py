import numpy as np

real = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
age = []

def real_age():
    for i in range(11):
        if real[i] == 0:
            print("11-15세")
        elif real[i] == 1:
            print("16-20세")
        elif real[i] == 2:
            print("21-25세")
        elif real[i] == 3:
            print("26-30세")
        elif real[i] == 4:
            print("31-35세")
        elif real[i] == 5:
            print("36-40세")
        elif real[i] == 6:
            print("41-45세")
        elif real[i] == 7:
            print("46-50세")
        elif real[i] == 8:
            print("51-55세")
        elif real[i] == 9:
            print("56-60세")
        elif real[i] == 10:
            print("61세 이상")
        else:
            print("ERR")
        age.append(real[i])
    return print(age)

real_age()