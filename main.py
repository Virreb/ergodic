import graph, ant_colony_optimization as aco
import numpy as np
from joblib import Parallel, delayed

# Set constants
nbr_cities = 10
nbr_transport_types = 5     # 0: bicycle, 1: car, 2:buss, 3: boat, 4: plane
nbr_parallell_colonies = 4

# Init map
city_locations = graph.generate_random_locations(nbr_cities)
distance_matrix = graph.create_distance_matrix(city_locations)
city_extra_points = graph.generate_random_city_extra_points(nbr_cities, max_point=1)

# Create graph
static_connection_graph = graph.generate_static_connection_graph(nbr_transport_types, nbr_cities)
punishment_graph = graph.add_random_paths_to_static_graph(distance_matrix, static_connection_graph)
print(f'{np.count_nonzero(punishment_graph > 0)} / {nbr_cities*nbr_cities*nbr_transport_types} connections')
#travel_time_graph, punishment_graph, score_graph = graph.generate_punishment_graph_from_distance(nbr_transport_types, distance_matrix, city_extra_points)

# Generate result
result = Parallel(n_jobs=4)(delayed(aco.summon_the_ergodic_colony)(punishment_graph, city_extra_points,
                                                                   start_city=0, target_city=9, nbr_ants=50)
                            for i in range(nbr_parallell_colonies))
print(result)

#aco.summon_the_ergodic_colony(punishment_graph, city_extra_points, start_city=0, target_city=9, nbr_ants=50)
