import tkinter as tk
import pandas as pd
import numpy as np
import copy, random, math
import ast
from tkinter import ttk
import psutil
import time
import folium
import webbrowser
import os
import seaborn as sns
from geopy.distance import geodesic
from folium.plugins import MarkerCluster
import matplotlib

matplotlib.use('TkAgg')  # set the backend

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from UIproblem2 import generate_random_solution, objective_function, midpoint_crossover, randompoint_crossover, mutate_solution_1, mutate_solution_2, generate_population, print_population, tournament_select, get_greatest_fit, replace_least_fittest, roulette_select, genetic_algorithm
#from UIproblem2 import get_sa_solution, get_neighbor_solution1, get_neighbor_solution2, get_neighbor_solution3, get_tabu_search_solution



root = tk.Tk()
root.title("Vehicle Routing Problem")




# Create a label with a big description
description_label = tk.Label(root, text="This is ASAE vehicle routing problem project.\n Developed by:\n Farzam Salimi\n Luis Henriques")
description_label.grid(row=0, column=0, columnspan=4)

#####here we add our genetic algorithm objective_function
############################################################

distances = pd.read_csv('distances.csv')
establishments = pd.read_csv('establishments.csv')

# shift the column names to the left
distances.columns = pd.Index(list(distances.columns[1:]) + ['p_1000'])

very_small_establishments = establishments.iloc[0:21,:]
very_small_distances = distances.iloc[0:21,0:21]

small_establishments = establishments.iloc[0:101,:]
small_distances = distances.iloc[0:101,0:101]

medium_establishments = establishments.iloc[0:501,:]
medium_distances = distances.iloc[0:501,0:501]

large_establishments = establishments
large_distances = distances





#####################################################################
###################repeated functions######################



#objective function with greedy search
def objective_function_greedy(chromosome, traveling_times, inspection_times, open_hours, heuristic_distances):
    total_travel_time = 0
    total_inspection_time = 0
    total_time_per_route = [0 for i in range(max(chromosome))]
    j = -1

    # Initialize an empty list to store the routes
    routes = [[] for i in range(max(chromosome))]
    best_routes = []

    # Assign each establishment to a route
    for i in range(len(chromosome)):
        routes[chromosome[i]-1].append(i+1)

    # Calculate the total travel time and inspection time for all routes
    for route in routes:
        if len(route) == 0:
            continue
        j += 1
        route_travel_time = 0
        route_inspection_time = 0
        current_time = 0
        # Add the departure/arrival establishment to the beginning and end of the route
        route = [0] + route + [0]

        # Use a greedy search algorithm to find the optimal path for the current route
        route = greedy_search(route, heuristic_distances)
        best_routes.append(route)

        start_time = ast.literal_eval(open_hours.iloc[route[1]][0]).index(1)  # Inspection start time of first establishment in route
        #print('start time: {}'.format(start_time))

        for i in range(1, len(route)):
            # Calculate traveling time between establishments
            current_establishment = route[i-1]
            next_establishment = route[i]
            travel_time = traveling_times[current_establishment][next_establishment]
            #route_travel_time += traveling_times[current_establishment][next_establishment]

            # Calculate inspection time and waiting time based on establishment's schedule
            current_inspection_time = inspection_times.iloc[next_establishment][0]*60 # Convert minutes to seconds
            current_open_hours = ast.literal_eval(open_hours.iloc[next_establishment][0])

            # Calculate waiting time if establishment is closed
            if i == 1:
                waiting_time = 0
            else:
                current_hour = int((current_time + travel_time + (start_time*3600))/3600)
                #print('Current hour: {}'.format(current_hour))
                while current_open_hours[current_hour % 24] == 0:
                    current_hour += 1
                waiting_time = (current_hour * 3600) - (current_time + travel_time + (start_time*3600))
                #print('({} * 3600) - ({} + {} + ({}*3600))'.format(current_hour, current_time, travel_time, start_time))

            # Add inspection time and waiting time to current time
            if i > 1:
                current_time += max(waiting_time, 0)
            current_time += travel_time
            current_time += current_inspection_time

            #print('Iteration {} ---> current establishment: {}, next establishment: {}, inspection time: {}, open hours array: {}, travel time: {}, waiting time: {}, current time: {}'.format(i, current_establishment, next_establishment, current_inspection_time, current_open_hours, travel_time, waiting_time, current_time))
            # Reset waiting time
            waiting_time = 0

            # Add inspection time and traveling time to route times
            #route_inspection_time += current_inspection_time
            route_travel_time = 0

        # Add total time for current route to total time per route
        total_time_per_route[j] = current_time
        #print('total time per route ----> {}'.format(total_time_per_route))

    # Add total time for all routes to total travel time
    total_travel_time += sum(total_time_per_route)

    return -total_travel_time, best_routes # Minimize the sum of inspection and travel time


#objective function with A-star
def objective_function_a_star(chromosome, traveling_times, inspection_times, open_hours, heuristic_distances):

    total_travel_time = 0
    l = 1

    # Initialize an empty list to store the routes
    routes = [[] for i in range(max(chromosome))]
    best_routes = []

    # Assign each establishment to a route
    for i in range(len(chromosome)):
        routes[chromosome[i]-1].append(i+1)

    # Calculate the total travel time for all routes
    for route in routes:
        if len(route) == 0:
            continue
        j = 1
        current_time = 0

        # Create a set with all establishments in the route
        route_establishments = set(route)
        route_len = len(route_establishments)

        # Add the depot to the beginning of the route
        route = [0]

        # Initialize the route travel time as zero
        route_travel_time = 0

        while route_establishments:
            # Calculate the A* algorithm for the next establishment
            current_establishment = route[-1]
            next_establishment = None
            best_score = float('inf')
            for e in route_establishments:
                i = 1
                # Calculate the total time (travel time, inspection time, waiting time, heuristic value) for the next establishment
                travel_time = traveling_times[route[-1]][e]
                inspection_time = inspection_times.iloc[e][0] * 60
                current_open_hours = ast.literal_eval(open_hours.iloc[e][0])
                if len(route_establishments) == route_len:
                    start_time = current_open_hours.index(1)
                i += 1
                # Calculate waiting time if establishment is closed
                if len(route) == 1:
                    waiting_time = 0
                else:
                    current_hour = int((current_time + travel_time + (start_time*3600))/3600)
                    while current_open_hours[current_hour % 24] == 0:
                        current_hour += 1
                    waiting_time = (current_hour * 3600) - (current_time + travel_time + (start_time*3600))

                # Add inspection time, waiting time, and travel time to the current time
                time_to_next = max(waiting_time, 0) + inspection_time + travel_time
                total_time = current_time + time_to_next
                heuristic_value = heuristic_distances[e][0]
                score = total_time + heuristic_value

                # If the current establishment has the smallest score, update next_establishment
                if score < best_score:
                    '''
                    best_waiting_time = max(waiting_time, 0)
                    best_open_hours = current_open_hours
                    best_inspection_time = inspection_time
                    if len(route) == 1:
                        best_start_time = start_time
                    else:
                        best_current_hour = current_hour
                    best_current_time = total_time
                    best_travel_time = travel_time
                    '''
                    time_to_next_establishment = time_to_next
                    next_establishment = e
                    best_score = score
            '''
            if len(route) > 1:
                print('Current hour: {}'.format(best_current_hour))
                print('({} * 3600) - ({} + {} + ({}*3600))'.format(best_current_hour, current_time, best_travel_time, best_start_time))
            else:
                print('start time: {}'.format(best_start_time))
            '''

            # Add next establishment to route and remove from set
            route_establishments.remove(next_establishment)
            route.append(next_establishment)

            #print('Update current time: {} + {}'.format(current_time, time_to_next_establishment))
            # Update the current time and route travel time
            current_time += time_to_next_establishment
            #route_travel_time += travel_time
            #print('Iteration {} ---> current establishment: {}, next establishment: {}, next inspection time: {}, open hours array: {}, travel time: {}, waiting time: {}, current time: {}'.format(j, current_establishment, next_establishment, best_inspection_time, best_open_hours, best_travel_time, best_waiting_time, best_current_time))

            j += 1
        # Add the depot to the end of the route
        route.append(0)
        best_routes.append(route)
        #print('Route: {}'.format(route))
        # Calculate the total time for the current route and add to total time per route
        total_time_per_route = current_time + traveling_times[route[-2]][0]
        #print('Traveling time back to the depot: {}'.format(traveling_times[route[-2]][0]))
        #print('Total travel time in route {}: {}'.format(l, total_time_per_route))
        total_travel_time += total_time_per_route
        l += 1

    return -total_travel_time, best_routes # Minimize the total travel time

def geodesic_time_matrix(establishments, speed_kmph):
    """
    Calculates the travel time matrix between all combinations of establishments based on their
    latitude and longitude coordinates, assuming a given speed in km/h.

    Parameters:
    establishments (pandas.DataFrame): A DataFrame containing the latitude and longitude coordinates of all establishments.
    speed_kmph (float): The speed of travel in km/h.

    Returns:
    pandas.DataFrame: A DataFrame containing the travel time matrix between all combinations of establishments in seconds.
    """
    # Create a 2D array of establishment coordinates
    coordinates = establishments[['Latitude', 'Longitude']].values

    # Initialize an empty time matrix with the same shape as the coordinates matrix
    time_matrix = pd.DataFrame(index=establishments.index, columns=establishments.index)

    # Calculate pairwise travel times between all establishments
    for i in range(len(coordinates)):
        for j in range(len(coordinates)):
            # Calculate the distance between establishments i and j using geodesic distance
            distance_km = geodesic(coordinates[i], coordinates[j]).km

            # Calculate the travel time in seconds using the speed in km/h
            travel_time_sec = (distance_km / speed_kmph) * 3600

            # Set the travel time in the time matrix
            time_matrix.iloc[i, j] = travel_time_sec

    return time_matrix

def greedy_search(route, heuristic):
    current_node = 0
    visited = [False] * (max(route) + 1)
    visited[current_node] = True
    path = [current_node]
    while len(path) < len(route):
        next_node = -1
        min_distance = float('inf')
        for i in range(len(route)):
            if not visited[route[i]]:
                distance = heuristic[route[i]][route[0]]
                if distance < min_distance:
                    next_node = route[i]
                    min_distance = distance
        visited[next_node] = True
        path.append(next_node)
        current_node = next_node
    path.pop() # remove last element (should be 0)
    path.append(0)
    return path



#function to display maps inside UI
def display_routes(routes):
    # Define the map center and zoom level
    center = [41.160304, -8.602478]
    zoom = 10.4

    # Create a map object
    tile = 'OpenStreetMap'
    map = folium.Map(location=center, zoom_start=zoom, tiles=tile)

    # Define the color palette for the routes
    route_colors = ['#d62728','#3388ff','#33cc33','#ff9933','#800080','#ff3399','#808080','#f5f5dc','#32174d','#ffffff',
              '#5f9ea0','#d3d3d3','#add8e6','#00008b','#90ee90','#006400','#8b0000','#ff6666']
    marker_colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'gray', 'beige', 'darkpurple', 'white',
                     'cadetblue', 'lightgray', 'lightblue', 'darkblue', 'lightgreen', 'darkgreen', 'darkred', 'lightred']

    # Create checkboxes for each route
    checkboxes = []
    for i, route in enumerate(routes):
        label = f"Route {i+1}"
        checkbox = folium.FeatureGroup(name=label, overlay=True, control=True)
        for k, j in enumerate(route[1:-1]):
            folium.Marker(
                location=(establishments.iloc[j]['Latitude'], establishments.iloc[j]['Longitude']),
                icon=folium.Icon(color=marker_colors[i % len(route_colors)]),
                tooltip=f"Visit order: {k+1}",
                popup=f"{establishments.iloc[j]['Address']}",
            ).add_to(checkbox)
        folium.PolyLine(
            locations=[(establishments.iloc[j]['Latitude'], establishments.iloc[j]['Longitude']) for j in route],
            color=route_colors[i % len(route_colors)],
            popup=label,
        ).add_to(checkbox)
        checkboxes.append(checkbox)

    # Add caution marker to the depot
    folium.Marker(location=center, icon=folium.Icon(icon='home', prefix='fa', color='black'), tooltip='Depot',
                 popup=f"{establishments.iloc[0]['Address']}").add_to(map)

    # Add the checkboxes to the map
    for checkbox in checkboxes:
        checkbox.add_to(map)

    # Add a layer control to the map
    folium.LayerControl().add_to(map)

    # Return the map object
    return map


def open_map():
    new_window = tk.Toplevel()
    new_window.title("Map")
    new_window.geometry("600x600")
    map_canvas = tk.Canvas(new_window, width=600, height=400)
    map_canvas.grid()
    map_canvas.create_text(300, 200, text='Loading map...', font=("Arial", 12))
    webbrowser.open_new_tab('file://' + os.path.abspath('map.html'))


###############################Genetic Algorithm section of the UI################################
def calculate_objective():
    # Get the selected size of establishments from the dropdown menu
    selected_size = size_var1.get()
    selected_cross = size_var6.get()
    selected_algorithm = heuristic_var1.get()
    num_iterations1 = int(num_iter_entry1.get())
    pop_size1 = int(num_iter_entry11.get())

    # Retrieve the corresponding establishments and distances dataframes
    if selected_size == "Very Small":
        establishments = pd.read_csv('establishments.csv').iloc[0:21,:]
        distances = pd.read_csv('distances.csv').iloc[0:21,0:21]
    elif selected_size == "Small":
        establishments = pd.read_csv('establishments.csv').iloc[0:101,:]
        distances = pd.read_csv('distances.csv').iloc[0:101,0:101]
    elif selected_size == "Medium":
        establishments = pd.read_csv('establishments.csv').iloc[0:501,:]
        distances = pd.read_csv('distances.csv').iloc[0:501,0:501]
    elif selected_size == "Large":
        establishments = pd.read_csv('establishments.csv')
        distances = pd.read_csv('distances.csv')

    # Convert the DataFrame to a dictionary
    traveling_times = {}
    for i in range(len(distances)):
        traveling_times[i] = distances.iloc[i].tolist()

    num_establishments = len(traveling_times)-1
    vehicles = int(num_establishments*0.1)
    inspection_times = establishments[['Inspection Time']]
    open_hours = establishments[['Opening Hours']]


    def generate_random_solution():
        return np.random.randint(1, vehicles + 1, num_establishments)




    heuristic_distances = geodesic_time_matrix(establishments, 70)

        # Subtract the heuristic distance matrix from the travel time matrix
    difference_matrix = heuristic_distances - distances


    def midpoint_crossover(solution_1, solution_2):
        mid_index = math.trunc(len(solution_1) / 2)
        child_1 = np.append(solution_1[0:mid_index], solution_2[mid_index:])
        child_2 = np.append(solution_2[0:mid_index], solution_1[mid_index:])
        return child_1, child_2

    def randompoint_crossover(solution_1, solution_2):
        length = len(solution_1)
        crossover_point = random.randint(1, length - 1)
        child_1 = np.concatenate([solution_1[:crossover_point], solution_2[crossover_point:]])
        child_2 = np.concatenate([solution_2[:crossover_point], solution_1[crossover_point:]])
        return child_1, child_2

    def uniform_crossover(solution_1, solution_2):
        """
        Performs uniform crossover (UX) on two solutions.

        Args:
            solution_1 (list): The first solution.
            solution_2 (list): The second solution.

        Returns:
            Two new solutions created by randomly swapping genes between solution_1 and solution_2.
        """
        # Initialize children as copies of the parent solutions
        child_1 = solution_1.copy()
        child_2 = solution_2.copy()

        # Determine the genes to swap
        genes_to_swap = np.random.choice(len(solution_1), len(solution_1), replace=False)

        # Swap the genes between the two children
        for i in genes_to_swap:
            if child_1[i] != child_2[i]:
                child_1[i], child_2[i] = child_2[i], child_1[i]

        return child_1, child_2

    def mutate_solution_1(solution):
        index_1 = np.random.randint(0, len(solution))
        index_2 = (index_1 + np.random.randint(0, len(solution))) % (len(solution) - 1) # Efficient way to generate a non-repeated index
        solution[index_1], solution[index_2] = solution[index_2], solution[index_1]
        return solution

    def mutate_solution_2(solution):
        index = np.random.randint(0, len(solution))
        solution[index] = np.random.randint(1, vehicles + 1)
        return solution

    def generate_population(population_size):
        solutions = []
        for i in range(population_size):
            solutions.append(generate_random_solution())
        return solutions

    def print_population(population, objective_func):
        solutions = []
        for i in range(len(population)):
            print(f"Solution {i+1}: {population[i]}, {-objective_func(population[i], traveling_times, inspection_times, open_hours, heuristic_distances)[0]}")

    def tournament_select(population, tournament_size, objective_func):
        population_copy = copy.deepcopy(population)
        best_solution = population_copy[0]
        best_score = objective_func(population_copy[0], traveling_times, inspection_times, open_hours, heuristic_distances)[0]
        for i in range(tournament_size):
            index = np.random.randint(0, len(population_copy))
            score = objective_func(population_copy[index], traveling_times, inspection_times, open_hours, heuristic_distances)[0]
            if score > best_score:
                best_score = score
                best_solution = population_copy[index]
            del population_copy[index]
        return best_solution

    def get_greatest_fit(population, objective_func):
        best_solution = population[0]
        best_score, best_routes = objective_func(population[0], traveling_times, inspection_times, open_hours, heuristic_distances)
        for i in range(1, len(population)):
            score, routes = objective_func(population[i], traveling_times, inspection_times, open_hours, heuristic_distances)
            if score > best_score:
                best_score = score
                best_solution = population[i]
                best_routes = routes
        return best_solution, best_score, best_routes

    def replace_least_fittest(population, offspring, objective_func):
        least_fittest_index = 0
        least_fittest_value = objective_func(population[0], traveling_times, inspection_times, open_hours, heuristic_distances)[0]
        for i in range(1, len(population)):
            score = objective_func(population[i], traveling_times, inspection_times, open_hours, heuristic_distances)[0]
            if score < least_fittest_value:
                least_fittest_value = score
                least_fittest_index = i
        population[least_fittest_index] = offspring

    def roulette_select(population, objective_func):
        score_sum = sum([objective_func(solution, traveling_times, inspection_times, open_hours, heuristic_distances)[0] for solution in population])
        selection_probabilities = [objective_func(solution, traveling_times, inspection_times, open_hours, heuristic_distances)[0] / score_sum for solution in population]
        return population[np.random.choice(len(population), p=selection_probabilities)]








    def genetic_algorithm(num_iterations, population_size, crossover_func, mutation_func, objective_func, log=False):

        start_time = time.time()
        log_messages = ""
        score_array = []
        iteration=0;

        population = generate_population(population_size)


        best_solution = population[0] # Initial solution
        best_score = objective_func(population[0], traveling_times, inspection_times, open_hours, heuristic_distances)[0]
        best_solution_generation = 0 # Generation on which the best solution was found

        generation_no = 0

        print(f"Initial solution: {best_solution}, score: {-best_score}")

        while(num_iterations > 0):
            mem_usage = psutil.virtual_memory().percent

            generation_no += 1

            tournment_winner_sol = tournament_select(population, 4, objective_func)
            roulette_winner_sol = roulette_select(population, objective_func)

            # Next generation
            crossover_sol_1, crossover_sol_2 = crossover_func(tournment_winner_sol, roulette_winner_sol)

            if np.random.randint(0, 10) < 1: # 40% chance to perform mutation
                replace_least_fittest(population, mutation_func(crossover_sol_1), objective_func)
                replace_least_fittest(population, mutation_func(crossover_sol_2), objective_func)
            else:
                replace_least_fittest(population, crossover_sol_1, objective_func)
                replace_least_fittest(population, crossover_sol_2, objective_func)

            # Checking the greatest fit among the current population
            greatest_fit, greatest_fit_score, best_routes = get_greatest_fit(population, objective_func)
            if greatest_fit_score > best_score:
                best_solution = greatest_fit
                best_score = greatest_fit_score
                best_routes
                best_solution_generation = generation_no
                if log:
                    print(f"\nGeneration: {generation_no }")
                    print(f"Solution: score: {-best_score}")
                    print_population(population, objective_func)
            else:
                num_iterations -= 1


            score_array.append(greatest_fit_score)

            iteration += 1
            mem_usage_after = psutil.virtual_memory().percent
            if log:
                log_messages += f"Solution: {best_solution}, score: {-best_score}, , best routes: {best_routes}, Iteration {iteration}: Memory usage: {mem_usage}%, Memory usage after processing: {mem_usage_after}%\n"


        sns.set_style("whitegrid")
        plt.plot(score_array)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.show()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f"  Final solution: {best_solution}, score: {-best_score}")
        print(f"  Found on generation {best_solution_generation}")
        print(f"  Final solution: {best_solution}, score: {-best_score}, best routes: {best_routes}")



        return best_solution, log_messages, best_routes






    #def run_genetic_algorithm(selected_cross):
    if selected_cross == "Midpoint Crossover":
        crossover_func = midpoint_crossover
    elif selected_cross == "Randompoint Crossover":
        crossover_func = randompoint_crossover
    elif selected_cross == "UX Crossover":
        crossover_func = uniform_crossover



    if selected_algorithm == "Greedy algorithm":
        objective_function = objective_function_greedy
    elif selected_algorithm == "A-star algorithm":
        objective_function = objective_function_a_star




    objective, log_messages, best_routes = genetic_algorithm(num_iterations1, pop_size1, crossover_func, mutate_solution_1, objective_function, True)


    #run_genetic_algorithm(selected_cross)
    # Display the objective value in the label
    return objective, log_messages, best_routes



###########################Designing the look##################################


label1 = tk.Label(root, text="Genetic Algorithm\n Select size of establishments:")
label1.config(font=("Helvetica", 14), fg="white")
label1.grid(row=1, column=0)


# Create the dropdown menu
size_options1 = ["Very Small", "Small", "Medium", "Large"]
size_var1 = tk.StringVar(root)
size_var1.set(size_options1[0])
dropdown1 = ttk.Combobox(root, textvariable=size_var1, values=size_options1)
dropdown1.grid(row=2, column=0)


num_iter_label1 = tk.Label(root, text="Number of iterations:")
num_iter_entry1 = tk.Entry(root)
num_iter_label1.grid(row=3, column=0)
num_iter_entry1.grid(row=4, column=0)


num_iter_label11 = tk.Label(root, text="Population size:")
num_iter_entry11 = tk.Entry(root)
num_iter_label11.grid(row=5, column=0)
num_iter_entry11.grid(row=6, column=0)







num_iter_label111 = tk.Label(root, text="Crossover Method:")
num_iter_label111.grid(row=9, column=0)
size_options6 = ["Midpoint Crossover", "Randompoint Crossover", "UX Crossover"]
size_var6 = tk.StringVar(root)
size_var6.set(size_options6[0])
dropdown6 = ttk.Combobox(root, textvariable=size_var6, values=size_options6)
dropdown6.grid(row=10, column=0)




heur_label111 = tk.Label(root, text="Route opt method:")
heur_label111.grid(row=11, column=0)
heuristic_options1 = ["Greedy algorithm", "A-star algorithm"]
heuristic_var1 = tk.StringVar(root)
heuristic_var1.set(heuristic_options1[0])
dropdown10 = ttk.Combobox(root, textvariable=heuristic_var1, values=heuristic_options1)
dropdown10.grid(row=12, column=0)





def show_result1():
    print("Geting things ready, please wait...")
    objective, log_messages, best_routes = calculate_objective()

    # create new window
    new_window = tk.Toplevel()

    # set window title and size
    new_window.title("Results")
    new_window.geometry("600x600")

    # set font
    font = ("Arial", 12)

    # create label for objective value
    objective_label = tk.Label(new_window, text=f"Score value: {objective}", font=font)
    objective_label.grid()

    # create frame for map
    map_frame = tk.Frame(new_window, width=600, height=600)
    map_frame.grid(row=1)




    routes_map = display_routes(best_routes)
    routes_map.save('map.html')
    map_canvas = tk.Canvas(map_frame, width=600, height=40)
    map_canvas.grid()
    map_canvas.create_text(300, 20, text='Map is loading...', font=font)



    map_button = tk.Button(map_frame, text="Show Map", font=font, command=open_map)
    map_button.grid(row=2)


    # create frame for log messages
    log_frame = tk.Frame(new_window, relief="groove", borderwidth=2)
    log_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    # create text widget for log messages
    log_widget = tk.Text(log_frame, font=font, wrap="word")
    log_widget.insert(tk.END, log_messages)
    log_widget.grid(column=0, row=0, padx=5, pady=5, sticky="nsew")

    # create scrollbar for log messages
    scrollbar = tk.Scrollbar(log_frame, command=log_widget.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    log_widget.config(yscrollcommand=scrollbar.set)



    #display_routes(best_routes)


# Create the button to calculate the objective function
#calculate_button = tk.Button(root, text="Calculate Objective", command=show_result)
#calculate_button.pack()
button1 = tk.Button(root, text="Genetic Algorithm", command=show_result1)
button1.config( font=("Arial", 12), relief="raised", borderwidth=2)
button1.grid(row=13, column=0)







#########################################################



################################Simulated annealing part of the UI###############################
def calculate_objective2():
    # Get the selected size of establishments from the dropdown menu
    selected_size = size_var2.get()
    selected_neigh1 = size_var7.get()
    selected_algorithm2 = heuristic_var2.get()
    num_iterations2 = int(num_iter_entry2.get())
    temperature2 = int(num_iter_entry21.get())
    Coolingrate = float(num_iter_entry22.get())

    # Retrieve the corresponding establishments and distances dataframes
    if selected_size == "Very Small":
        establishments = pd.read_csv('establishments.csv').iloc[0:21,:]
        distances = pd.read_csv('distances.csv').iloc[0:21,0:21]
    elif selected_size == "Small":
        establishments = pd.read_csv('establishments.csv').iloc[0:101,:]
        distances = pd.read_csv('distances.csv').iloc[0:101,0:101]
    elif selected_size == "Medium":
        establishments = pd.read_csv('establishments.csv').iloc[0:501,:]
        distances = pd.read_csv('distances.csv').iloc[0:501,0:501]
    elif selected_size == "Large":
        establishments = pd.read_csv('establishments.csv')
        distances = pd.read_csv('distances.csv')

    # Convert the DataFrame to a dictionary
    traveling_times = {}
    for i in range(len(distances)):
        traveling_times[i] = distances.iloc[i].tolist()

    num_establishments = len(traveling_times)-1
    vehicles = int(num_establishments*0.1)
    inspection_times = establishments[['Inspection Time']]
    open_hours = establishments[['Opening Hours']]


    def generate_random_solution():
        return np.random.randint(1, vehicles + 1, num_establishments)



    heuristic_distances = geodesic_time_matrix(establishments, 70)
    difference_matrix = heuristic_distances - distances

    # Neighborhood definition

    def get_neighbor_solution1(solution):
        neighbor = copy.deepcopy(solution)
        establishment_to_mutate = np.random.randint(0, num_establishments)
        neighbor[establishment_to_mutate] = (neighbor[establishment_to_mutate] + np.random.randint(1, vehicles)-1) % vehicles + 1
        return neighbor

    # Exchange the vehicles of two establishments
    def get_neighbor_solution2(solution):
        neighbor_solution = copy.deepcopy(solution)
        establishment1 = np.random.randint(0, num_establishments)
        establishment2 = (establishment1 + np.random.randint(1, num_establishments)) % num_establishments
        neighbor_solution[establishment1], neighbor_solution[establishment2] = neighbor_solution[establishment2], neighbor_solution[establishment1]
        return neighbor_solution

    # Neighbour 1 or 2 with 50% each
    def get_neighbor_solution3(solution):
        if (np.random.randint(0,2)==0):
            return get_neighbor_solution1(solution)
        else:
            return get_neighbor_solution2(solution)


    def get_sa_solution(num_iterations, neighbor_operator, traveling_times, inspection_times, open_hours, objective_func,log=False):
        start_time2 = time.time()
        log_messages = ""  # Initialize the log_messages variable here
        iteration = 0;
        temperature = temperature2;
        score_array = []
        solution = generate_random_solution() # Best solution after 'num_iterations' iterations without improvement
        score, routes = objective_func(solution, traveling_times, inspection_times, open_hours, heuristic_distances)

        best_solution = copy.deepcopy(solution)
        best_score = score
        best_routes = routes

        print(f"Init Solution:  {best_solution}, score: {-best_score}")

        while iteration < num_iterations:
            mem_usage2 = psutil.virtual_memory().percent
            temperature = temperature * Coolingrate  # Test with different cooling schedules
            iteration += 1
            neighbor_solution = neighbor_operator(best_solution)  #Test with Neighbour 1, 2 and 3
            neighbor_score, neighbor_routes = objective_func(neighbor_solution, traveling_times, inspection_times, open_hours, heuristic_distances)

            delta = neighbor_score - score

            if delta > 0 or np.exp(-delta/temperature) > random.random():
                solution = neighbor_solution
                score = neighbor_score
                routes = neighbor_routes
                if score > best_score:
                    iteration = 0
                    best_solution = copy.deepcopy(solution)
                    best_score = score
                    best_routes = routes
                    if log:
                        print(f"Solution: score: {-best_score},  Temp: {temperature}")

            mem_usage_after2 = psutil.virtual_memory().percent
            if log:  # Use the log_messages variable here
                log_messages += f"Solution:       {best_solution}, score: {-best_score},  Temp: {temperature}\n, Iteration {iteration}: Memory usage: {mem_usage2}%, Memory usage after processing: {mem_usage_after2}%\n"
                # Store the current score for plotting

            score_array.append(score)
            mem_usage_after2 = psutil.virtual_memory().percent

        # Plot the scores
        end_time2 = time.time()
        elapsed_time2 = end_time2 - start_time2
        print(f"Elapsed time: {elapsed_time2} seconds")
        sns.set_style("whitegrid")
        plt.plot(score_array)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.show()
        return best_solution, log_messages, best_routes



    if selected_neigh1 == "Neighbour1":
        neighbour_func = get_neighbor_solution1
    elif selected_neigh1 == "Neighbour2":
        neighbour_func = get_neighbor_solution2
    elif selected_neigh1 == "Neighbour3":
        neighbour_func = get_neighbor_solution3



    if selected_algorithm2 == "Greedy algorithm":
        objective_function = objective_function_greedy
    elif selected_algorithm2 == "A-star algorithm":
        objective_function = objective_function_a_star

    objective, log_messages, best_routes = get_sa_solution(num_iterations2, neighbour_func, traveling_times, inspection_times, open_hours, objective_function, True)

    #print(f"Final Solution: {objective}, score: {-objective_function(objective, traveling_times, inspection_times, open_hours)}")

    return objective, log_messages, best_routes



#############################Designing the look################################
# Create a label for the dropdown menu
label2 = tk.Label(root, text="Simulated Annealing\n Select size of establishments:")
label2.config(font=("Helvetica", 14), fg="white")
label2.grid(row=1, column=1)


# Create the dropdown menu
size_options2 = ["Very Small", "Small", "Medium", "Large"]
size_var2 = tk.StringVar(root)
size_var2.set(size_options2[0])
dropdown2 = ttk.Combobox(root, textvariable=size_var2, values=size_options2)
dropdown2.grid(row=2, column=1)



num_iter_label2 = tk.Label(root, text="Number of iterations:")
num_iter_entry2 = tk.Entry(root)
num_iter_label2.grid(row=3, column=1)
num_iter_entry2.grid(row=4, column=1)


num_iter_label21 = tk.Label(root, text="Temperature:")
num_iter_entry21 = tk.Entry(root)
num_iter_label21.grid(row=5, column=1)
num_iter_entry21.grid(row=6, column=1)



num_iter_label22 = tk.Label(root, text="Cooling rate:")
num_iter_entry22 = tk.Entry(root)
num_iter_label22.grid(row=7, column=1)
num_iter_entry22.grid(row=8, column=1)







neib_label22 = tk.Label(root, text="Neighbouring method:")
neib_label22.grid(row=9, column=1)
size_options7 = ["Neighbour1", "Neighbour2", "Neighbour3"]
size_var7 = tk.StringVar(root)
size_var7.set(size_options7[0])
dropdown7 = ttk.Combobox(root, textvariable=size_var7, values=size_options7)
dropdown7.grid(row=10, column=1)




algo_label22 = tk.Label(root, text="Route opt method:")
algo_label22.grid(row=11, column=1)
heuristic_options2 = ["Greedy algorithm", "A-star algorithm"]
heuristic_var2 = tk.StringVar(root)
heuristic_var2.set(heuristic_options2[0])
dropdown11 = ttk.Combobox(root, textvariable=heuristic_var2, values=heuristic_options2)
dropdown11.grid(row=12, column=1)




def show_result2():
    print("Geting things ready, please wait...")
    objective, log_messages, best_routes = calculate_objective2()

    # create new window
    new_window = tk.Toplevel()

    # set window title and size
    new_window.title("Results")
    new_window.geometry("600x600")

    # set font
    font = ("Arial", 12)

    # create label for objective value
    objective_label = tk.Label(new_window, text=f"Score value: {objective}", font=font)
    objective_label.grid()

    # create frame for map
    map_frame = tk.Frame(new_window, width=600, height=600)
    map_frame.grid(row=1)




    routes_map = display_routes(best_routes)
    routes_map.save('map.html')
    map_canvas = tk.Canvas(map_frame, width=600, height=40)
    map_canvas.grid()
    map_canvas.create_text(300, 20, text='Map is loading...', font=font)



    map_button = tk.Button(map_frame, text="Show Map", font=font, command=open_map)
    map_button.grid(row=2)


    # create frame for log messages
    log_frame = tk.Frame(new_window, relief="groove", borderwidth=2)
    log_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    # create text widget for log messages
    log_widget = tk.Text(log_frame, font=font, wrap="word")
    log_widget.insert(tk.END, log_messages)
    log_widget.grid(column=0, row=0, padx=5, pady=5, sticky="nsew")

    # create scrollbar for log messages
    scrollbar = tk.Scrollbar(log_frame, command=log_widget.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    log_widget.config(yscrollcommand=scrollbar.set)


    #calculate_button = tk.Button(root, text="Calculate Objective", command=show_result)
#calculate_button.pack()
button2 = tk.Button(root, text="Simulate Annealing", command=show_result2)
button2.config(font=("Arial", 12), relief="raised", borderwidth=2)
button2.grid(row=13, column=1)



###################################Tabu Search section############################################



def calculate_objective3():

    # Get the selected size of establishments from the dropdown menu
    selected_size = size_var3.get()
    selected_algorithm3 = heuristic_var3.get()
    num_iterations3 = int(num_iter_entry3.get())
    tabu_tenur3 = int(num_iter_entry31.get())
    selected_neigh2 = size_var8.get()

    # Retrieve the corresponding establishments and distances dataframes
    if selected_size == "Very Small":
        establishments = pd.read_csv('establishments.csv').iloc[0:21,:]
        distances = pd.read_csv('distances.csv').iloc[0:21,0:21]
    elif selected_size == "Small":
        establishments = pd.read_csv('establishments.csv').iloc[0:101,:]
        distances = pd.read_csv('distances.csv').iloc[0:101,0:101]
    elif selected_size == "Medium":
        establishments = pd.read_csv('establishments.csv').iloc[0:501,:]
        distances = pd.read_csv('distances.csv').iloc[0:501,0:501]
    elif selected_size == "Large":
        establishments = pd.read_csv('establishments.csv')
        distances = pd.read_csv('distances.csv')

    # Convert the DataFrame to a dictionary
    traveling_times = {}
    for i in range(len(distances)):
        traveling_times[i] = distances.iloc[i].tolist()

    num_establishments = len(traveling_times)-1
    vehicles = int(num_establishments*0.1)
    inspection_times = establishments[['Inspection Time']]
    open_hours = establishments[['Opening Hours']]


    # Calculate the objective function using the selected traveling times
    def generate_random_solution3():
        return np.random.randint(1, vehicles + 1, num_establishments)



    heuristic_distances = geodesic_time_matrix(establishments, 70)
    difference_matrix = heuristic_distances - distances


        # Neighborhood definition

    # Neighborhood definition

    def get_neighbor_solution1(solution):
        neighbor = copy.deepcopy(solution)
        establishment_to_mutate = np.random.randint(0, num_establishments)
        neighbor[establishment_to_mutate] = (neighbor[establishment_to_mutate] + np.random.randint(1, vehicles)-1) % vehicles +1
        return neighbor

    # Exchange the vehicles of two establishments
    def get_neighbor_solution2(solution):
        neighbor_solution = copy.deepcopy(solution)
        establishment1 = np.random.randint(0, num_establishments)
        establishment2 = (establishment1 + np.random.randint(1, num_establishments)) % num_establishments
        neighbor_solution[establishment1], neighbor_solution[establishment2] = neighbor_solution[establishment2], neighbor_solution[establishment1]
        return neighbor_solution

    # Neighbour 1 or 2 with 50% each
    def get_neighbor_solution3(solution):
        if (np.random.randint(0,2)==0):
            return get_neighbor_solution1(solution)
        else:
            return get_neighbor_solution2(solution)



    def get_tabu_search_solution(num_iterations, tabu_list_length, neighbor_operator, traveling_times, inspection_times, open_hours, objective_func,log=False):
        start_time3 = time.time()
        log_messages = ""
        score_array = []
        iteration = 0;
        current_solution = generate_random_solution3()
        current_score, current_route = objective_func(current_solution, traveling_times, inspection_times, open_hours, heuristic_distances)

        best_solution = current_solution
        best_score = current_score
        best_route = current_route

        tabu_list = []

        for iteration in range(num_iterations):
            mem_usage3 = psutil.virtual_memory().percent
            neighbor_solutions = []
            neighbor_scores = []
            neighbor_routes = []
            tabu_neighbors = []

            for i in range(10):
                neighbor_solution = neighbor_operator(current_solution)
                neighbor_score, neighbor_route = objective_func(neighbor_solution, traveling_times, inspection_times, open_hours, heuristic_distances)
                if neighbor_solution.tolist() not in tabu_list:
                    neighbor_solutions.append(neighbor_solution)
                    neighbor_scores.append(neighbor_score)
                    neighbor_routes.append(neighbor_route)
                else:
                    tabu_neighbors.append(neighbor_solution)

            if len(neighbor_scores) > 0:
                best_neighbor_index = np.argmax(neighbor_scores)
                best_neighbor_score = neighbor_scores[best_neighbor_index]
                best_neighbor_solution = neighbor_solutions[best_neighbor_index]
                best_neighbor_route = neighbor_routes[best_neighbor_index]

                if best_neighbor_score > best_score:
                    best_solution = best_neighbor_solution
                    best_score = best_neighbor_score
                    best_route = best_neighbor_route

                current_solution = best_neighbor_solution
                current_score = best_neighbor_score
                current_route = best_neighbor_route

                tabu_list.append(current_solution.tolist())
                if len(tabu_list) > tabu_list_length:
                    tabu_list.pop(0)

            elif len(tabu_neighbors) > 0:
                best_tabu_index = np.argmax([objective_func(x, traveling_times, inspection_times, open_hours)[0] for x in tabu_neighbors])
                best_tabu_solution = tabu_neighbors[best_tabu_index]

                current_solution = best_tabu_solution
                current_score, current_route = objective_func(current_solution, traveling_times, inspection_times, open_hours, heuristic_distances)

                tabu_list.append(current_solution.tolist())
                if len(tabu_list) > tabu_list_length:
                    tabu_list.pop(0)

            score_array.append(current_score)

            mem_usage_after3 = psutil.virtual_memory().percent

            if log:
                log_messages += f"Solution:       {best_solution},Iteration {iteration}, Best Score: {-best_score}\n, Iteration {iteration}: Memory usage: {mem_usage3}%, Memory usage after processing: {mem_usage_after3}%\n"

            iteration += 1

        end_time3 = time.time()
        sns.set_style("whitegrid")
        plt.plot(score_array)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.show()
        elapsed_time3 = end_time3 - start_time3
        print(f"Elapsed time: {elapsed_time3} seconds")
        return best_solution, log_messages, best_route




    if selected_neigh2 == "Neighbour1":
        neighbour_func2 = get_neighbor_solution1
    elif selected_neigh2 == "Neighbour2":
        neighbour_func2 = get_neighbor_solution2
    elif selected_neigh2 == "Neighbour3":
        neighbour_func2 = get_neighbor_solution3


    if selected_algorithm3 == "Greedy algorithm":
        objective_function = objective_function_greedy
    elif selected_algorithm3 == "A-star algorithm":
        objective_function = objective_function_a_star



    objective, log_messages, best_route = get_tabu_search_solution(num_iterations3, tabu_tenur3, neighbour_func2, traveling_times, inspection_times, open_hours, objective_function, True)
    return objective, log_messages, best_route


    # Display the objective value in the label
    #objective_label.config(text=f"Objective value: {objective}")



#############################################################


label3 = tk.Label(root, text="Tabu Search\n Select size of establishments:")
label3.grid(row=1, column=2)

size_options3 = ["Very Small", "Small", "Medium", "Large"]
size_var3 = tk.StringVar(root)
size_var3.set(size_options3[0])
dropdown3 = ttk.Combobox(root, textvariable=size_var3, values=size_options3)
dropdown3.grid(row=2, column=2)



num_iter_label3 = tk.Label(root, text="Number of iterations:")
num_iter_entry3 = tk.Entry(root)
num_iter_label3.grid(row=3, column=2)
num_iter_entry3.grid(row=4, column=2)



num_iter_label31 = tk.Label(root, text="Tabu tenure:")
num_iter_entry31 = tk.Entry(root)
num_iter_label31.grid(row=5, column=2)
num_iter_entry31.grid(row=6, column=2)




neighb_label31 = tk.Label(root, text="Neighbouring method:")
neighb_label31.grid(row=9, column=2)
size_options8 = ["Neighbour1", "Neighbour2", "Neighbour3"]
size_var8 = tk.StringVar(root)
size_var8.set(size_options8[0])
dropdown8 = ttk.Combobox(root, textvariable=size_var8, values=size_options8)
dropdown8.grid(row=10, column=2)




algo_label31 = tk.Label(root, text="Route opt method:")
algo_label31.grid(row=11, column=2)
heuristic_options3 = ["Greedy algorithm", "A-star algorithm"]
heuristic_var3 = tk.StringVar(root)
heuristic_var3.set(heuristic_options3[0])
dropdown11 = ttk.Combobox(root, textvariable=heuristic_var3, values=heuristic_options3)
dropdown11.grid(row=12, column=2)




def show_result3():
    print("Geting things ready, please wait...")
    objective, log_messages, best_routes = calculate_objective3()

    # create new window
    new_window = tk.Toplevel()

    # set window title and size
    new_window.title("Results")
    new_window.geometry("600x600")

    # set font
    font = ("Arial", 12)

    # create label for objective value
    objective_label = tk.Label(new_window, text=f"Score value: {objective}", font=font)
    objective_label.grid()

    # create frame for map
    map_frame = tk.Frame(new_window, width=600, height=600)
    map_frame.grid(row=1)




    routes_map = display_routes(best_routes)
    routes_map.save('map.html')
    map_canvas = tk.Canvas(map_frame, width=600, height=40)
    map_canvas.grid()
    map_canvas.create_text(300, 20, text='Map is loading...', font=font)



    map_button = tk.Button(map_frame, text="Show Map", font=font, command=open_map)
    map_button.grid(row=2)


    # create frame for log messages
    log_frame = tk.Frame(new_window, relief="groove", borderwidth=2)
    log_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    # create text widget for log messages
    log_widget = tk.Text(log_frame, font=font, wrap="word")
    log_widget.insert(tk.END, log_messages)
    log_widget.grid(column=0, row=0, padx=5, pady=5, sticky="nsew")

    # create scrollbar for log messages
    scrollbar = tk.Scrollbar(log_frame, command=log_widget.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    log_widget.config(yscrollcommand=scrollbar.set)



# Create the button to calculate the objective function
button3 = tk.Button(root, text="Tabu Search", command=show_result3)
button3.grid(row=13, column=2)

####################################Hill climbing###########################################


def calculate_objective4():

    # Get the selected size of establishments from the dropdown menu
    selected_size = size_var4.get()
    selected_algorithm4 = heuristic_var4.get()
    num_iterations4 = int(num_iter_entry4.get())
    selected_neigh3 = size_var9.get()

    # Retrieve the corresponding establishments and distances dataframes
    if selected_size == "Very Small":
        establishments = pd.read_csv('establishments.csv').iloc[0:21,:]
        distances = pd.read_csv('distances.csv').iloc[0:21,0:21]
    elif selected_size == "Small":
        establishments = pd.read_csv('establishments.csv').iloc[0:101,:]
        distances = pd.read_csv('distances.csv').iloc[0:101,0:101]
    elif selected_size == "Medium":
        establishments = pd.read_csv('establishments.csv').iloc[0:501,:]
        distances = pd.read_csv('distances.csv').iloc[0:501,0:501]
    elif selected_size == "Large":
        establishments = pd.read_csv('establishments.csv')
        distances = pd.read_csv('distances.csv')

    # Convert the DataFrame to a dictionary
    traveling_times = {}
    for i in range(len(distances)):
        traveling_times[i] = distances.iloc[i].tolist()

    num_establishments = len(traveling_times)-1
    vehicles = int(num_establishments*0.1)
    inspection_times = establishments[['Inspection Time']]
    open_hours = establishments[['Opening Hours']]


    # Calculate the objective function using the selected traveling times
    def generate_random_solution():
        return np.random.randint(1, vehicles + 1, num_establishments)



    heuristic_distances = geodesic_time_matrix(establishments, 70)
    difference_matrix = heuristic_distances - distances

    # Neighborhood definition

    def get_neighbor_solution1(solution):
        neighbor = copy.deepcopy(solution)
        establishment_to_mutate = np.random.randint(0, num_establishments)
        neighbor[establishment_to_mutate] = (neighbor[establishment_to_mutate] + np.random.randint(1, vehicles)-1) % vehicles + 1
        return neighbor

    # Exchange the vehicles of two establishments
    def get_neighbor_solution2(solution):
        neighbor_solution = copy.deepcopy(solution)
        establishment1 = np.random.randint(0, num_establishments)
        establishment2 = (establishment1 + np.random.randint(1, num_establishments)) % num_establishments
        neighbor_solution[establishment1], neighbor_solution[establishment2] = neighbor_solution[establishment2], neighbor_solution[establishment1]
        return neighbor_solution

    # Neighbour 1 or 2 with 50% each
    def get_neighbor_solution3(solution):
        if (np.random.randint(0,2)==0):
            return get_neighbor_solution1(solution)
        else:
            return get_neighbor_solution2(solution)



    def get_hc_solution(num_iterations, neighbor_operator, traveling_times, inspection_times, open_hours, objective_func, log=False):
        iteration = 0;
        log_messages = ""
        score_array = []
        start_time3 = time.time()
        best_solution = generate_random_solution() # Best solution after 'num_iterations' iterations without improvement
        best_score, best_routes = objective_func(best_solution, traveling_times, inspection_times, open_hours, heuristic_distances)

        print(f"Init Solution: score: {-best_score}")

        while iteration < num_iterations:
            iteration += 1
            neighbor_solution = neighbor_operator(best_solution)   #Test with Neighbour 1, 2 and 3
            neighbor_score, neighbor_routes = objective_func(neighbor_solution, traveling_times, inspection_times, open_hours, heuristic_distances)
            if neighbor_score > best_score:
                iteration = 0
                best_solution = neighbor_solution
                best_score = neighbor_score
                best_routes = neighbor_routes
                if log:
                    (print(f"Solution {iteration}: score: {-best_score}"))

            score_array.append(best_score)

            iteration += 1

            if log:
                log_messages += f"Solution {iteration}: score: {-best_score}"

        end_time3 = time.time()
        sns.set_style("whitegrid")
        plt.plot(score_array)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.show()
        elapsed_time3 = end_time3 - start_time3
        print(f"Elapsed time: {elapsed_time3} seconds")
        return best_solution, log_messages, best_routes



    if selected_neigh3 == "Neighbour1":
        neighbour_func3 = get_neighbor_solution1
    elif selected_neigh3 == "Neighbour2":
        neighbour_func3 = get_neighbor_solution2
    elif selected_neigh3 == "Neighbour3":
        neighbour_func3 = get_neighbor_solution3



    if selected_algorithm4 == "Greedy algorithm":
        objective_function = objective_function_greedy
    elif selected_algorithm4 == "A-star algorithm":
        objective_function = objective_function_a_star

    objective, log_messages, best_routes = get_hc_solution(num_iterations4, neighbour_func3, traveling_times, inspection_times, open_hours, objective_function, True)
    return objective, log_messages, best_routes


    # Display the objective value in the label
    #objective_label.config(text=f"Objective value: {objective}")



#############################################################


label4 = tk.Label(root, text="Hill Climbing\n Select size of establishments:")
label4.grid(row=1, column=3)

size_options4 = ["Very Small", "Small", "Medium", "Large"]
size_var4 = tk.StringVar(root)
size_var4.set(size_options4[0])
dropdown4 = ttk.Combobox(root, textvariable=size_var4, values=size_options4)
dropdown4.grid(row=2, column=3)


num_iter_label4 = tk.Label(root, text="Number of iterations:")
num_iter_entry4 = tk.Entry(root)
num_iter_label4.grid(row=3, column=3)
num_iter_entry4.grid(row=4, column=3)



neighb_label4 = tk.Label(root, text="Neighbouring method:")
neighb_label4.grid(row=9, column=3)
size_options9 = ["Neighbour1", "Neighbour2", "Neighbour3"]
size_var9 = tk.StringVar(root)
size_var9.set(size_options9[0])
dropdown9 = ttk.Combobox(root, textvariable=size_var9, values=size_options9)
dropdown9.grid(row=10, column=3)




algo_label4 = tk.Label(root, text="Route opt method:")
algo_label4.grid(row=11, column=3)
heuristic_options4 = ["Greedy algorithm", "A-star algorithm"]
heuristic_var4 = tk.StringVar(root)
heuristic_var4.set(heuristic_options1[0])
dropdown12 = ttk.Combobox(root, textvariable=heuristic_var4, values=heuristic_options4)
dropdown12.grid(row=12, column=3)


def show_result4():
    print("Geting things ready, please wait...")
    objective, log_messages, best_routes = calculate_objective4()

    # create new window
    new_window = tk.Toplevel()

    # set window title and size
    new_window.title("Results")
    new_window.geometry("600x600")

    # set font
    font = ("Arial", 12)

    # create label for objective value
    objective_label = tk.Label(new_window, text=f"Score value: {objective}", font=font)
    objective_label.grid()

    # create frame for map
    map_frame = tk.Frame(new_window, width=600, height=600)
    map_frame.grid(row=1)




    routes_map = display_routes(best_routes)
    routes_map.save('map.html')
    map_canvas = tk.Canvas(map_frame, width=600, height=40)
    map_canvas.grid()
    map_canvas.create_text(300, 20, text='Map is loading...', font=font)



    map_button = tk.Button(map_frame, text="Show Map", font=font, command=open_map)
    map_button.grid(row=2)


    # create frame for log messages
    log_frame = tk.Frame(new_window, relief="groove", borderwidth=2)
    log_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    # create text widget for log messages
    log_widget = tk.Text(log_frame, font=font, wrap="word")
    log_widget.insert(tk.END, log_messages)
    log_widget.grid(column=0, row=0, padx=5, pady=5, sticky="nsew")

    # create scrollbar for log messages
    scrollbar = tk.Scrollbar(log_frame, command=log_widget.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    log_widget.config(yscrollcommand=scrollbar.set)



# Create the button to calculate the objective function
button4 = tk.Button(root, text="Hill Climbing", command=show_result4)
button4.grid(row=13, column=3)







##########################################################

root.mainloop()
