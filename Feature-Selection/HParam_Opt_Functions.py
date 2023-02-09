import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import svm
from copy import deepcopy



def objective_value(chromosome, data):
    # original data
    x_org_data = data.drop(["Y"],axis=1)
    y = pd.DataFrame(data,columns=["Y"]).values

    norm = preprocessing.MinMaxScaler()
    x = norm.fit_transform(x_org_data)

    # Reading Chromosome
    emp_list = [i for i,e in enumerate(chromosome) if e == 1]
    # To cover the case when chromoseme = [0, ... , 0]
    # and prevent program to crash
    if emp_list == []:
        return 1

    new_x = x[:,emp_list]

    kfold = 3
    kf = KFold(n_splits=kfold)
    # objective function value for the decoded x and decoded y
    sum_of_error = 0
    for train_index,test_index in kf.split(new_x):
        x_train,x_test = new_x[train_index],new_x[test_index]
        y_train,y_test = y[train_index],y[test_index]

        model = svm.SVC()
        model.fit(x_train,np.ravel(y_train))
        accuracy = model.score(x_test,y_test)
        error = 1-(accuracy)
        sum_of_error += error

    avg_error = sum_of_error/kfold

    return avg_error


def find_parents_ts(all_solutions):
    # Tournament Selection
    tournament = sorted(all_solutions[np.random.choice(all_solutions.shape[0], 3, replace=False)]
                        , key = lambda x: x[0])
    return tournament[0][1]


def crossover(parent_1,parent_2,prob_crsvr=1):
    crsvr_or_not = np.random.rand()
    if crsvr_or_not < prob_crsvr:
        index_1 = np.random.randint(0,len(parent_1))
        index_2 = np.random.randint(index_1,len(parent_1))

        first_seg_parent_1 = parent_1[:index_1]
        mid_seg_parent_1 = parent_1[index_1:index_2+1]
        last_seg_parent_1 = parent_1[index_2+1:]

        first_seg_parent_2 = parent_2[:index_1]
        mid_seg_parent_2 = parent_2[index_1:index_2+1]
        last_seg_parent_2 = parent_2[index_2+1:]


        ### CHILD_1 ###
        # parent_1 | parent_2 | parent_1
        child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                  last_seg_parent_1))
        ### CHILD_2 ###
        # parent_2 | parent_1 | parent_2
        child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                  last_seg_parent_2))

    else:
        child_1 = deepcopy(parent_1)
        child_2 = deepcopy(parent_2)

    return child_1,child_2


def mutation(child,prob_mutation=0.2):
    for index in range(len(child)):
        mutate_or_not = np.random.rand()
        if mutate_or_not < prob_mutation:
            child[index] = 1 - child[index] #flip gene