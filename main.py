import graph, ant_colony_optimization as aco

nbr_cities = 10
nbr_transport_types = 5     # 0: bicycle, 1: car, 2:buss, 3: boat, 4: plane

city_locations = graph.generate_random_locations(nbr_cities)
distance_matrix = graph.create_distance_matrix(city_locations)

travel_time_graph, punishment_graph, score_graph = graph.generate_punishment_graph_from_distance(nbr_transport_types, distance_matrix)

pheromones = aco.initiate_pheromones(score_graph)


#punishment_graph = init.generate_random_punishment_graph(nbr_transport_types, nbr_cities)
#print(punishment_graph)

#print(travel_time_graph)
#print('\n\n')
#print(punishment_graph)
#print('\n\n')
#print(visibility_graph)

