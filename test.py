import numpy as np

nbr_of_cities = 10
nbr_of_transports = 2
nbr_ants = 30

# Initiate graph with random punishment 0-49
travel_graph = np.full(shape=(nbr_of_transports, nbr_of_cities, nbr_of_cities), fill_value=0)
travel_graph += np.random.randint(50, size=(nbr_of_transports, nbr_of_cities, nbr_of_cities))

travel_graph[travel_graph == 0] = None
print(travel_graph)

print(travel_graph[0, 1, 1])
start_city = np.random.randint(nbr_of_cities-1) + 1

target_city = start_city
while target_city != start_city:
    target_city = np.random.randint(nbr_of_cities-1) + 1


#visited_cities = []
#for k_ant in range(nbr_ants):
#    print(k_ant)












