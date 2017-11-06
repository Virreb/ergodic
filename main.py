import graph, ant_colony_optimization as aco
import numpy as np
from joblib import Parallel, delayed

# Set constants
nbr_cities = 10
nbr_transport_types = 5     # 0: bicycle, 1: car, 2:buss, 3: boat, 4: plane
nbr_colonies = 8
nbr_parallel_jobs = 4

# Init map
city_locations = graph.generate_random_locations(nbr_cities)
distance_matrix = graph.create_distance_matrix(city_locations)
city_extra_points = graph.generate_random_city_extra_points(nbr_cities, max_point=1)

# Create graph
static_connection_graph = graph.generate_static_connection_graph(nbr_transport_types, nbr_cities)
punishment_graph = graph.add_random_paths_to_static_graph(distance_matrix, static_connection_graph)
punishment_matrix, transport_matrix = graph.merge_graph_to_matrix(punishment_graph)

print(f'{np.count_nonzero(~np.isnan(punishment_matrix))} / {nbr_cities*nbr_cities} connections')
#travel_time_graph, punishment_graph, score_graph = graph.generate_punishment_graph_from_distance(nbr_transport_types, distance_matrix, city_extra_points)

# Set arguments
args = (punishment_matrix, transport_matrix, city_extra_points)
kwargs = {
    'start_city': 2,
    'target_city': 9,
    'nbr_ants': 50,
    'verbose': True,
    'evaporation': 0.5,
    'alpha': 1.0,   # pheromones
    'beta': 3.0     # scores
}

# Run colonies in parallel
best_path, best_score, all_results = aco.run_parallel_colonies(nbr_parallel_jobs, nbr_colonies, args, kwargs)

# Run 1 colony
#best_path, best_score = aco.summon_the_ergodic_colony(*args, **kwargs)

# Plot the graph
graph.plot_graph(city_locations, punishment_graph, best_path)
