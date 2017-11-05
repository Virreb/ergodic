import graph, ant_colony_optimization as aco
import numpy as np

nbr_cities = 10
nbr_transport_types = 5     # 0: bicycle, 1: car, 2:buss, 3: boat, 4: plane

city_locations = graph.generate_random_locations(nbr_cities)
distance_matrix = graph.create_distance_matrix(city_locations)
city_extra_points = graph.generate_random_city_extra_points(nbr_cities, max_point=1)

static_connection_graph = graph.generate_static_connection_graph(nbr_transport_types, nbr_cities)
punishment_graph = graph.add_random_paths_to_static_graph(distance_matrix, static_connection_graph)

#print(punishment_graph)
print(f'{np.count_nonzero(punishment_graph > 0)} / {nbr_cities*nbr_cities*nbr_transport_types} connections')

#travel_time_graph, punishment_graph, score_graph = graph.generate_punishment_graph_from_distance(nbr_transport_types, distance_matrix, city_extra_points)

aco.summon_the_ergodic_colony(punishment_graph, city_extra_points, start_city=0, target_city=9, nbr_ants=50)

#punishment_graph = init.generate_random_punishment_graph(nbr_transport_types, nbr_cities)
#print(punishment_graph)

#print(travel_time_graph)
#print('\n\n')
#print(punishment_graph)
#print('\n\n')
#print(visibility_graph)

