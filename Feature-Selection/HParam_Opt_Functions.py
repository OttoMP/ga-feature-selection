import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from copy import deepcopy



# calculate fitness value for the chromosome of 0s and 1s
def objective_value(x,y,chromosome,kfold=3):
    emp_list = [i for i,e in enumerate(chromosome) if e == 1]

    # To cover the case when chromoseme = [0, ... , 0]
    # and prevent program to crash
    if emp_list == []:
        return 1

    new_x = x[:,emp_list]

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


# finding 2 parents from the pool of solutions
# using the tournament selection method
def find_parents_ts(all_solutions,x,y):

    # select 3 random parents from the pool of solutions you have
    posb_parents = all_solutions[np.random.choice(all_solutions.shape[0], 3, replace=False)]

    # get objective function value (fitness) for each possible parent
    # index no.2 because the objective_value function gives the fitness value at index no.2
    tournament = sorted([(parent, objective_value(x,y,parent)) for parent in posb_parents], key = lambda x: x[1])

    return tournament[0][0]


# crossover between the 2 parents to create 2 children
# functions inputs are parent_1, parent_2, and the probability you would like for crossover
# default probability of crossover is 1
def crossover(parent_1,parent_2,prob_crsvr=1):

    rand_num_to_crsvr_or_not = np.random.rand() # do we crossover or not???
    if rand_num_to_crsvr_or_not < prob_crsvr:
        index_1 = np.random.randint(0,len(parent_1))
        index_2 = np.random.randint(index_1,len(parent_1))

        ### FOR PARENT_1 ###

        # first_seg_parent_1 -->
        # for parent_1: the genes from the beginning of parent_1 to the
                # beginning of the middle segment of parent_1
        first_seg_parent_1 = parent_1[:index_1]

        # middle segment; where the crossover will happen
        # for parent_1: the genes from the index chosen for parent_1 to
                # the index chosen for parent_2
        mid_seg_parent_1 = parent_1[index_1:index_2+1]

        # last_seg_parent_1 -->
        # for parent_1: the genes from the end of the middle segment of
                # parent_1 to the last gene of parent_1
        last_seg_parent_1 = parent_1[index_2+1:]


        ### FOR PARENT_2 ###

        # first_seg_parent_2 --> same as parent_1
        first_seg_parent_2 = parent_2[:index_1]

        # mid_seg_parent_2 --> same as parent_1
        mid_seg_parent_2 = parent_2[index_1:index_2+1]

        # last_seg_parent_2 --> same as parent_1
        last_seg_parent_2 = parent_2[index_2+1:]


        ### CREATING CHILD_1 ###

        # the first segment from parent_1
        # plus the middle segment from parent_2
        # plus the last segment from parent_1
        child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                  last_seg_parent_1))


        ### CREATING CHILD_2 ###

        # the first segmant from parent_2
        # plus the middle segment from parent_1
        # plus the last segment from parent_2
        child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                  last_seg_parent_2))


    # when we will not crossover
    # when rand_num_to_crsvr_or_not is NOT less (is greater) than prob_crsvr
    # when prob_crsvr == 1, then rand_num_to_crsvr_or_not will always be less
            # than prob_crsvr, so we will always crossover then
    else:
        child_1 = deepcopy(parent_1)
        child_2 = deepcopy(parent_2)

    return child_1,child_2 # the defined function will return 2 arrays



############################################################
### MUTATING THE TWO CHILDREN TO CREATE MUTATED CHILDREN ###
############################################################

# mutation for the 2 children
# functions inputs are child_1, child_2, and the probability you would like for mutation
# default probability of mutation is 0.2
def mutation(child_1,prob_mutation=0.2):

    for index in range(len(child_1)): # for each gene (index)

        rand_num_to_mutate_or_not = np.random.rand() # do we mutate or no???
        # if the rand_num_to_mutate_or_not is less that the probability of mutation
                # then we mutate at that given gene (index we are currently at)
        if rand_num_to_mutate_or_not < prob_mutation:
            child_1[index] = 1 - child_1[index] #flip gene