import numpy as np
import pandas as pd
#import time
import GA_aux_func as hp_opt

def main():
    # Loading the data, shuffling and preprocessing it
    # Feature Selection
    data = pd.read_csv("../Dataset/breast-cancer-wisconsin.csv")
    data = data.sample(frac=1)

    n_samples = data.shape[0]
    n_features = data.shape[1]-1
    print("Number of samples:", n_samples)
    print("Number of features:", n_features)
    print("-----------------------------------")


    # Hyperparameters (user inputted parameters)
    #-------------------------------------------
    prob_crsvr = 1 # probablity of crossover
    prob_mutation = 0.2 # probablity of mutation
    population = 10 # population number
    generations = 2 # generation number

    # chromosome encoding
    # one gene for each feature
    chromosome_size = n_features

    # create an empty array to store a solution from each generation
    # for each generation, we want to save the best solution in that generation
    # to compare with the convergence of the algorithm
    best_of_a_generation = []

    # so now, pool_of_solutions, has n (population) chromosomes
    # chromosome = np.array([0,1,0,0,0,1,0,0,1,0,0,1,
    #                        0,1,1,1,0,0,1,0,1,1,1,0]) # initial solution
    pool_of_solutions = []
    for _ in range(population):
        parent = np.random.randint(2, size=chromosome_size)
        fitness = hp_opt.objective_value(parent, data)
        pool_of_solutions.append((fitness, parent))
    pool_of_solutions = np.array(pool_of_solutions, dtype = object) # Convert for easier manipulation

    # Start of Algorithm
    # ------------------
    for gen in range(generations):
        print("--> Generation: #", gen+1)
        print("-----------------------------------")

        # an empty array for saving the new generation
        # at the beginning of each generation, the array should be empty
        # so that you put all the solutions created in a certain generation
        new_population = []

        for family in range(int(population/2)): # population/2 because each gives 2 parents
            print("--> Family: #", family+1)
            print("-----------------------------------")

            # selecting 2 parents using tournament selection
            parent_1 = hp_opt.find_parents_ts(pool_of_solutions)
            parent_2 = hp_opt.find_parents_ts(pool_of_solutions)

            # crossover the 2 parents to get 2 children
            child_1, child_2 = hp_opt.crossover(parent_1, parent_2
                                                , prob_crsvr=prob_crsvr)

        # mutating the 2 children to get 2 mutated children
        # mutating the 2 children to get 2 mutated children
        # "genf.mutation"[0] gives mutated_child_1
        # "genf.mutation"[1] gives mutated_child_2
            # mutating the 2 children to get 2 mutated children
        # "genf.mutation"[0] gives mutated_child_1
        # "genf.mutation"[1] gives mutated_child_2
            hp_opt.mutation(child_1
                            , prob_mutation=prob_mutation)
            hp_opt.mutation(child_2
                            , prob_mutation=prob_mutation)


            # getting the obj val (fitness value) for the 2 mutated children
            obj_val_mutated_child_1 = hp_opt.objective_value(chromosome=child_1
                                                             , data=data)
            obj_val_mutated_child_2 = hp_opt.objective_value(chromosome=child_2
                                                             , data=data)


            # for each mutated child, put its obj val next to it
            mutant_1_with_obj_val = (obj_val_mutated_child_1, child_1)

            mutant_2_with_obj_val = (obj_val_mutated_child_2, child_2)


            new_population.append(mutant_1_with_obj_val)
            new_population.append(mutant_2_with_obj_val)


        # we replace the initial (before) population with the new one (current generation)
        # this new pool of solutions becomes the starting population of the next generation
        pool_of_solutions = np.array(new_population, dtype=object)
        best_of_a_generation.append(sorted(new_population,key=lambda x: x[0])[0])

    print(best_of_a_generation)
    best_last_generation = best_of_a_generation[-1]
    print("best last")
    print(best_last_generation)
    best_of_all = sorted(best_of_a_generation, key=lambda x:x[0])[0]
    print("best all")
    print(best_of_all)

    # since we sorted them from best to worst
    # the best would be the first solution in the array
    # so index [0] of the "sorted_last_population" array
    best_string_convergence = best_last_generation[1]
    emp_list_convergence = [i for i,e in enumerate(best_string_convergence) if e == 1]

    best_string_overall = best_of_all[1]
    emp_list_overall = [i for i,e in enumerate(best_string_overall) if e == 1]

    print("\n------------------------------\n")
    print("Final Solution (Convergence):",best_last_generation[1]) # final solution entire chromosome
    print("Obj Value - Convergence:",best_last_generation[0]) # obj val of final chromosome
    print("Features included are:",emp_list_convergence)
    print()
    print("Final Solution (Best):",best_of_all[1]) # final solution entire chromosome
    print("Obj Value - Best in Generations:",best_of_all[0]) # obj val of final chromosome
    print("Features included are:",emp_list_overall)

    # to decode the x and y chromosomes to their real values
    #final_solution_convergence = hp_opt.objective_value(chromosome=best_string_convergence
                                                        #, data=data)

    #final_solution_overall = hp_opt.objective_value(chromosome=best_string_overall
                                                    #, data=data)
    print("------------------------------")

if __name__ == "__main__":
    main()