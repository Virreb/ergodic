import numpy as np

nbr_of_cities = 10
city_points = np.random.randint(50, size=nbr_of_cities)
nbr_of_transports = 2

graph_weights_punnishment = np.random.randint(50, size=(nbr_of_transports,nbr_of_cities,nbr_of_cities))

visited_cities = []
print(graph_weights_punnishment)


start_city = 0
target_city = np.random.randint(nbr_of_cities-1) + 1

nbr_of_ants = 30
for k_ant in range(nbr_of_ants):
    print(k_ant)












