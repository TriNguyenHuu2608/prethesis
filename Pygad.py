import numpy as np
import pygad
import pygad.nn
import pygad.gann
import os 
from Data_set import Data_set

def fitness_func(solution, sol_idx):
    global GANN_instance, data_inputs, data_outputs

    predictions = pygad.nn.predict(last_layer=GANN_instance.population_networks[sol_idx],
                                    data_inputs=data_inputs)
                                    # ,problem_type="regression")
    # print('predictions: ', predictions)
    correct_predictions = np.where(predictions == data_outputs)[0].size
    solution_fitness = (correct_predictions/data_outputs.size)*100

    return solution_fitness

def callback_generation(ga_instance):
    global GANN_instance
    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga_instance.population)                             
    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Accuracy   = {fitness}".format(fitness=ga_instance.best_solution()[1]))


if __name__ == '__main__':
    # data_inputs = numpy.array([[1, 1],[1, 0],[0, 1],[0, 0]])
    # print('data_inputs: ', data_inputs)
    # data_outputs = numpy.array([0, 1,1, 0]) 
    name="NH3"
    q = Data_set(name)
    data_inputs,data_outputs = q.get_data_ga()

    GANN_instance = pygad.gann.GANN(num_solutions=5,num_neurons_input=352,num_neurons_output=2,num_neurons_hidden_layers=[2],hidden_activations=["relu"],output_activation='softmax')
    
    population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)
    
    ga_instance = pygad.GA(num_generations=100, 
                        num_parents_mating=3, 
                        initial_population=population_vectors.copy(),
                        fitness_func=fitness_func,
                        mutation_percent_genes=5,
                        callback_generation=callback_generation)


    ga_instance.run()
    ga_instance.plot_result()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(solution)
    print(solution_fitness)
    print(solution_idx)