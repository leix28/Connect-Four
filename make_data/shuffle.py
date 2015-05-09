import random

if __name__ == "__main__":
    f = open('result/train_data.txt', 'r')

    lines = f.readlines()
    random.shuffle(lines)

    for line in lines:
        print line[:-1]
