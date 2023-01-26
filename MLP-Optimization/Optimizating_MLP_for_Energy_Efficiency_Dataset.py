import numpy as np
import pandas as pd
import random as rd
#import time
from sklearn import preprocessing
import MLP_HParam_Opt_Functions as mlp_hp_opt


# Loading the data, shuffling and preprocessing it
data = pd.read_csv("../Dataset/energy_efficiency.csv")
data = data.sample(frac=1)

# original data
x_org_data = pd.DataFrame(data,columns=["X1","X2","X3","X4",
                                        "X5","X6","X7","X8"])
y1 = pd.DataFrame(data, columns = ['Y1']).values
y2 = pd.DataFrame(data, columns = ['Y2']).values

x_with_dummies = pd.get_dummies(x_org_data,columns=["X6","X8"])
var_prep = preprocessing.MinMaxScaler()
x = var_prep.fit_transform(x_with_dummies)

y = y1[:,0]

# Hyperparameters (user inputted parameters)
#-------------------------------------------
prob_crsvr = 1 # probablity of crossover
prob_crsvr_int = 0.3 # Probability of crossover for integers
prob_mutation = 0.3 # probablity of mutation
prob_mutation_int = 0.2 # Probability of mutation for integers
population = 40 # population number
generations = 20 # generation number
kfold = 3


### Combinatorial ###
# Upper and Lower bounds for combinatorial variables
UB_X1 = 10 # X1, Number of Neurons
LB_X1 = 6
UB_X2 = 8 # X2, Number of Hidden Layers
LB_X2 = 3


### Continuous ###
# Where the first 15 represent X3 and the second 15 represent X4
# x and y decision variables' encoding
# 12 genes for x and 12 genes for y (arbitrary number)
x_var_size = 15
y_var_size = 15
chromosome_size = x_var_size+y_var_size

# create an empty array to store a solution from each generation
# for each generation, we want to save the best solution in that generation
# to compare with the convergence of the algorithm
best_of_a_generation = np.empty((0,chromosome_size+3))

# so now, pool_of_solutions, has n (population) chromosomes
# chromosome = np.array([0,1,0,0,0,1,0,0,1,0,0,1,
#                        0,1,1,1,0,0,1,0,1,1,1,0]) # initial solution
gene_pool = np.array([np.random.randint(2, size=chromosome_size) for _ in range(population)])
pool_of_solutions = np.empty((0, chromosome_size+2))
#TODO add solver type to gene
#Solver_Type = ['adam']
for i, gene in enumerate(gene_pool): # Shuffles the elements in the vector n times and stores them
    #ST = rd.choice(Solver_Type)
    X1 = rd.randrange(6,10,2)
    X2 = rd.randrange(3,8,1)
    pool_of_solutions = np.vstack((pool_of_solutions, np.insert(gene, 0, (X1,X2))))
    #n_list_ST = np.append(n_list_ST,ST)

#start_time = time.time() # start time (timing purposes)

for gen in range(generations): # do it n (generation) times

    # an empty array for saving the new generation
    # at the beginning of each generation, the array should be empty
    # so that you put all the solutions created in a certain generation
    new_population = np.empty((0,chromosome_size+2))

    # an empty array for saving the new generation plus its obj func val
    new_population_with_obj_val = np.empty((0,chromosome_size+3))

    # an empty array for saving the best solution (chromosome)
    # for each generation
    sorted_best = np.empty((0,chromosome_size+3))

    print("--> Generation: #", gen+1) # tracking purposes
    print("-----------------------------------")

    for family in range(int(population/2)): # population/2 because each gives 2 parents

        print("--> Family: #", family+1) # tracking purposes
        print("-----------------------------------")

        # selecting 2 parents using tournament selection
        # "genf.find_parents_ts"[0] gives parent_1
        # "genf.find_parents_ts"[1] gives parent_2
        parent_1 = mlp_hp_opt.find_parents_ts(pool_of_solutions,
                                              x=x,y=y)
        parent_2 = mlp_hp_opt.find_parents_ts(pool_of_solutions,
                                              x=x,y=y)

        # crossover the 2 parents to get 2 children
        # "genf.crossover"[0] gives child_1
        # "genf.crossover"[1] gives child_2
        child_1 = mlp_hp_opt.crossover(parent_1,parent_2,
                               prob_crsvr=prob_crsvr)[0]
        child_2 = mlp_hp_opt.crossover(parent_1,parent_2,
                               prob_crsvr=prob_crsvr)[1]


        # mutating the 2 children to get 2 mutated children
        # "genf.mutation"[0] gives mutated_child_1
        # "genf.mutation"[1] gives mutated_child_2
        mlp_hp_opt.mutation(child_1
                            , prob_mutation=prob_mutation
                            , prob_mutation_int= prob_mutation_int
                            , ub_x1=UB_X1, lb_x1=LB_X1
                            , ub_x2=UB_X2, lb_x2=LB_X2)
        mlp_hp_opt.mutation(child_2
                            , prob_mutation=prob_mutation
                            , prob_mutation_int= prob_mutation_int
                            , ub_x1=UB_X1, lb_x1=LB_X1
                            , ub_x2=UB_X2, lb_x2=LB_X2)


        # getting the obj val (fitness value) for the 2 mutated children
        # "genf.objective_value"[2] gives obj val for the mutated child
        obj_val_mutated_child_1 = mlp_hp_opt.objective_value(x=x,y=y,
                                                             solution=child_1,
                                                             kfold=kfold)[2]
        obj_val_mutated_child_2 = mlp_hp_opt.objective_value(x=x,y=y,
                                                             solution=child_2,
                                                             kfold=kfold)[2]


        # for each mutated child, put its obj val next to it
        mutant_1_with_obj_val = np.hstack((obj_val_mutated_child_1,
                                               child_1)) # lines 132 and 140

        mutant_2_with_obj_val = np.hstack((obj_val_mutated_child_2,
                                               child_2)) # lines 134 and 143


        # we need to create the new population for the next generation
        # so for each family, we get 2 solutions
        # we keep on adding them till we are done with all the families in one generation
        # by the end of each generation, we should have the same number as the initial population
        # so this keeps on growing and growing
        # when it's a new generation, this array empties and we start the stacking process
        # and so on
        # check line 88
        new_population = np.vstack((new_population,
                                    child_1,
                                    child_2))


        # same explanation as above, but we include the obj val for each solution as well
        # check line 91
        new_population_with_obj_val = np.vstack((new_population_with_obj_val,
                                                 mutant_1_with_obj_val,
                                                 mutant_2_with_obj_val))
        # after getting 2 mutated children (solutions), we get another 2, and so on
        # until we have the same number of the intended population
        # then we go to the next generation and start over
        # since we ended up with 2 solutions, we move on to the next possible solutions


    # we replace the initial (before) population with the new one (current generation)
    # this new pool of solutions becomes the starting population of the next generation
    pool_of_solutions = new_population


    # for each generation
    # we want to find the best solution in that generation
    # so we sort them based on index [0], which is the obj val
    sorted_best = np.array(sorted(new_population_with_obj_val,
                                               key=lambda x:x[0]))


    # since we sorted them from best to worst
    # the best in that generation would be the first solution in the array
    # so index [0] of the "sorted_best" array
    best_of_a_generation = np.vstack((best_of_a_generation,
                                      sorted_best[0]))


#end_time = time.time() # end time (timing purposes)


# for our very last generation, we have the last population
# for this array of last population (convergence), there is a best solution
# so we sort them from best to worst
sorted_last_population = np.array(sorted(new_population_with_obj_val,
                                         key=lambda x:x[0]))

sorted_best_of_a_generation = np.array(sorted(best_of_a_generation,
                                         key=lambda x:x[0]))

sorted_last_population[:,0] = 1-(sorted_last_population[:,0]) # get accuracy instead of error
sorted_best_of_a_generation[:,0] = 1-(sorted_best_of_a_generation[:,0])

# since we sorted them from best to worst
# the best would be the first solution in the array
# so index [0] of the "sorted_last_population" array
best_string_convergence = sorted_last_population[0]

best_string_overall = sorted_best_of_a_generation[0]

print()
#print()
#print("Execution Time in Minutes:",(end_time - start_time)/60) # exec. time


print()
print()
print("------------------------------")
print()
#print("Execution Time in Seconds:",end_time - start_time) # exec. time
#print()
print("Final Solution (Convergence):",best_string_convergence[1:]) # final solution entire chromosome
print("Encoded Learning Rate (Convergence):",best_string_convergence[3:3+chromosome_size+1]) # final solution x chromosome
print("Encoded Momentum (Convergence):",best_string_convergence[3+chromosome_size+1:]) # final solution y chromosome
print()
print("Final Solution (Best):",best_string_overall[1:]) # final solution entire chromosome
print("Encoded Learning Rate (Best):",best_string_overall[3:3+chromosome_size+1]) # final solution x chromosome
print("Encoded Momentum (Best):",best_string_overall[3+chromosome_size+1]) # final solution y chromosome

# to decode the x and y chromosomes to their real values
final_solution_convergence = mlp_hp_opt.objective_value(x=x,y=y,
                                                        solution=best_string_convergence[1:],
                                                        kfold=kfold)

final_solution_overall = mlp_hp_opt.objective_value(x=x,y=y,
                                                    solution=best_string_overall[1:],
                                                    kfold=kfold)

# the "svm_hp_opt.objective_value" function returns 3 things -->
# [0] is the x value
# [1] is the y value
# [2] is the obj val for the chromosome (avg. error)
print()
print("Decoded Learning Rate (Convergence):",round(final_solution_convergence[0],5)) # real value of x
print("Decoded Momentum (Convergence):",round(final_solution_convergence[1],5)) # real value of y
print("Obj Value - Convergence:",round(1-(final_solution_convergence[2]),5)) # obj val of final chromosome
print()
print("Decoded Leaning Rate (Best):",round(final_solution_overall[0],5)) # real value of x
print("Decoded Momentum (Best):",round(final_solution_overall[1],5)) # real value of y
print("Obj Value - Best in Generations:",round(1-(final_solution_overall[2]),5)) # obj val of final chromosome
print()
print("------------------------------")





