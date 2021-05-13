import numpy as np
import random


def data_split(fixing_split=False):
    if fixing_split == True:
        train_sample = [3, 4, 5, 9, 10, 12, 14 ,15, 16, 18, 20, 21, 25, 26, 27,29, 32, 36, 37, 38, 39, 41, 42, 43, 44,
                        45, 47, 48, 53, 54, 55, 56, 57, 60, 61, 65, 66, 67, 72, 73,77,78,74, 75, 76, 80, 81,
                        82, 83, 84, 86, 87, 88, 89, 91, 94, 96, 98, 99, 100]

        test_sample = [1, 2, 7, 13, 17, 19, 22, 23, 30, 31, 33, 34, 40, 46, 49, 51, 58, 59, 64, 68, 69,
                        71, 79, 92, 93] 
        val_sample = [6, 8, 11, 24, 28, 35, 50, 52, 62, 63, 70, 85, 90,95,97]
    # random split the datasets
    else:
        delete_sample = []
        ALL_list = list(np.arange(1,101))
        for i in delete_sample:
            ALL_list.remove(i)
        train_sample = []
        test_sample = []
        print("please input the num for training, testing, and evaluating(total<=100):")
        train_num = int(input())
        test_num = int(input())
        val_num = int(input())
        for i in range(train_num):
            random_choice = random.choice(ALL_list)
            train_sample.append(random_choice)
            ALL_list.remove(random_choice)

        for i in range(test_num):
            random_choice = random.choice(ALL_list)
            test_sample.append(random_choice)
            ALL_list.remove(random_choice)

        val_sample = ALL_list

        train_sample.sort()
        test_sample.sort()
        val_sample.sort()
    print("train:",train_sample,"---","test:",test_sample,"---","val:",val_sample)
    print('train_sample_num:',len(train_sample))
    print('test_sample_num:',len(test_sample))
    print('val_sample_num:',len(val_sample))
    return train_sample, test_sample, val_sample
