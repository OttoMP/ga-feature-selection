import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from copy import deepcopy



# calculate fitness value for the chromosome of 0s and 1s
def objective_value(x,y,solution,kfold=3):

    chromosome = solution[2:]
    #print("Chromosome", chromosome)
    # x = c
    lb_x = 10 # lower bound for chromosome x
    ub_x = 1000 # upper bound for chromosome x
    len_x = (len(chromosome)//2) # length of chromosome x

    # y = gamma
    lb_y = 0.05 # lower bound for chromosome y
    ub_y = 0.99 # upper bound for chromosome y
    len_y = (len(chromosome)//2) # length of chromosome y

    precision_x = (ub_x-lb_x)/((2**len_x)-1) # precision for decoding x
    precision_y = (ub_y-lb_y)/((2**len_y)-1) # precision for decoding y

    #z = 0 # because we start at 2^0, in the formula
    #t = 1 # because we start at the very last element of the vector [index -1]
    x_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for i in range(len(chromosome)//2):
        x_bit_sum += chromosome[-(i+1)]*(2**i)

    #z = 0 # because we start at 2^0, in the formula
    #t = 1 + (len(chromosome)//2) # [6,8,3,9] (first 2 are y, so index will be 1+2 = -3)
    y_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for j in range(len(chromosome)//2):
        y_bit_sum += chromosome[-(j + 1 + (len(chromosome)//2))]*(2**j)

    # the formulas to decode the chromosome of 0s and 1s to an actual number, the value of x or y
    learning_rate = (x_bit_sum*precision_x)+lb_x # Decoded X3
    momentum = (y_bit_sum*precision_y)+lb_y # Decoded X4

    #learning_rate = 0.014690694906460768
    #momentum =  0.3528968169194617
    #print("Learning Rate", learning_rate)
    #print("Momentum", momentum)

    kf = KFold(n_splits=kfold)
    comb1 = int(solution[0])
    #print("Comb 1", comb1)
    comb2 = int(solution[1])
    #print("Comb 2", comb2)
    # objective function value for the decoded x and decoded y
    sum_of_error = 0
    #print("------------------------")
    for train_index,test_index in kf.split(x):
        x_train,x_test = x[train_index],x[test_index]
        y_train,y_test = y[train_index],y[test_index]

        Hid_Lay = [comb1 for _ in range(comb2)]
        #print("Hidden Layer", Hid_Lay)
        model1 = MLPRegressor(activation='relu',hidden_layer_sizes=Hid_Lay,
                            learning_rate_init=learning_rate,momentum=momentum)

        model1.fit(x_train, y_train)
        prediction=model1.predict(x_test)
        accuracy=model1.score(x_test,y_test)
        #print("Accuracy (score)", accuracy)

        error = 1-accuracy
        #print("Error", error)
        sum_of_error += error
        #print("------------------------")
        #print("Sum of error", sum_of_error)
        #print("------------------------")

    avg_error = sum_of_error/kfold
    #print("Avg Error", avg_error)

    # the defined function will return 3 values
    return learning_rate,momentum,avg_error


# finding 2 parents from the pool of solutions
# using the tournament selection method
def find_parents_ts(all_solutions,x,y):

    # select 3 random parents from the pool of solutions you have
    posb_parents = all_solutions[np.random.choice(all_solutions.shape[0], 3, replace=False)]

    # get objective function value (fitness) for each possible parent
    # index no.2 because the objective_value function gives the fitness value at index no.2
    tournament = sorted([(parent, objective_value(x,y,parent)[2]) for parent in posb_parents], key = lambda x: x[1])

    return tournament[0][0]


# crossover between the 2 parents to create 2 children
# functions inputs are parent_1, parent_2, and the probability you would like for crossover
# default probability of crossover is 1
def crossover(parent_1,parent_2,prob_crsvr=1):
    # For X1
    rand_num_to_crsvr_or_not = np.random.rand()
    if rand_num_to_crsvr_or_not < prob_crsvr:
        int_X1_P1 = parent_2[0]
        int_X1_P2 = parent_1[0]
    else:
        int_X1_P1 = parent_1[0]
        int_X1_P2 = parent_2[0]

    # For X2
    rand_num_to_crsvr_or_not = np.random.rand()
    if rand_num_to_crsvr_or_not < prob_crsvr:
        int_X2_P1 = parent_2[1]
        int_X2_P2 = parent_1[1]
    else:
        int_X2_P1 = parent_1[1]
        int_X2_P2 = parent_2[1]

    rand_num_to_crsvr_or_not = np.random.rand() # do we crossover or not???
    if rand_num_to_crsvr_or_not < prob_crsvr:
        index_1 = np.random.randint(2,len(parent_1))
        index_2 = np.random.randint(index_1,len(parent_1))

        ### FOR PARENT_1 ###

        # first_seg_parent_1 -->
        # for parent_1: the genes from the beginning of parent_1 to the
                # beginning of the middle segment of parent_1
        first_seg_parent_1 = parent_1[2:index_1]

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
        first_seg_parent_2 = parent_2[2:index_1]

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

        child_1 = np.insert(child_1,0,(int_X1_P1,int_X2_P1))###
        child_2 = np.insert(child_2,0,(int_X1_P2,int_X2_P2))

    # when we will not crossover
    # when rand_num_to_crsvr_or_not is NOT less (is greater) than prob_crsvr
    # when prob_crsvr == 1, then rand_num_to_crsvr_or_not will always be less
            # than prob_crsvr, so we will always crossover then
    else:
        child_1 = deepcopy(parent_1[2:])
        child_2 = deepcopy(parent_2[2:])

        child_1 = np.insert(child_1,0,(int_X1_P1,int_X2_P1))###
        child_2 = np.insert(child_2,0,(int_X1_P2,int_X2_P2))

    return child_1,child_2 # the defined function will return 2 arrays



############################################################
### MUTATING THE TWO CHILDREN TO CREATE MUTATED CHILDREN ###
############################################################

# mutation for the 2 children
# functions inputs are child_1, child_2, and the probability you would like for mutation
# default probability of mutation is 0.2
def mutation(child,prob_mutation=0.1, prob_mutation_int=0.2
            , ub_x1=10, lb_x1=6, ub_x2=8, lb_x2=3):
    # For X1
    rand_num_to_mutate_or_not = np.random.rand()
    if rand_num_to_mutate_or_not < prob_mutation_int:
        up_or_down = np.random.rand()
        if up_or_down >= 0.5:
            if child[0] == ub_x1:
                c_X1 = child[0]
            elif child[0] == lb_x1:
                c_X1 = child[0]
            else:
                c_X1 = child[0] + 2
        else:
            if child[0] == ub_x1:
                c_X1 = child[0]
            elif child[0] == lb_x1:
                c_X1 = child[0]
            else:
                c_X1 = child[0] - 2
    else:
        c_X1 = child[0]

    # For X2
    rand_num_to_mutate_or_not = np.random.rand()
    if rand_num_to_mutate_or_not < prob_mutation_int:
        up_or_down = np.random.rand()
        if up_or_down >= 0.5:
            if child[1] == ub_x2:
                c_X2 = child[1]
            elif child[1] == lb_x2:
                c_X2 = child[1]
            else:
                c_X2 = child[1] + 1
        else:
            if child[1] == ub_x2:
                c_X2 = child[1]
            elif child[1] == lb_x2:
                c_X2 = child[1]
            else:
                c_X2 = child[1] - 1
    else:
        c_X2 = child[1]


    chromosome = child[2:]
    for index in range(len(chromosome)): # for each gene (index)

        rand_num_to_mutate_or_not = np.random.rand() # do we mutate or no???
        # if the rand_num_to_mutate_or_not is less that the probability of mutation
                # then we mutate at that given gene (index we are currently at)
        if rand_num_to_mutate_or_not < prob_mutation:
            chromosome[index] = 1 - chromosome[index] #flip gene

    child = np.insert(chromosome,0,(c_X1,c_X2))