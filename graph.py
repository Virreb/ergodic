import numpy as np


def generate_static_connection_graph(nbr_transport_types, nbr_cities):
    graph = np.nan * np.zeros(shape=(nbr_transport_types, nbr_cities, nbr_cities))
    connected_cities = [[0, 1], [0, 3], [0, 8],
                        [1, 0], [1, 9],
                        [2, 3], [2, 4], [2, 5],
                        [3, 2], [3, 0], [3, 8],
                        [4, 2], [4, 5],
                        [5, 2], [5, 7], [5, 6],
                        [6, 5], [6, 7],
                        [7, 5], [7, 6],
                        [8, 3], [8, 9], [8, 0],
                        [9, 8], [9, 1]]

    for connection in connected_cities:
        random_trsp = np.random.randint(nbr_transport_types)
        graph[random_trsp, connection[0], connection[1]] = 1

    return graph


def generate_random_locations(nbr_cities):
    import numpy as np

    max_len = 6000  # in km (Kiruna to Dubai)
    return np.random.randint(max_len, size=(nbr_cities, 2))


def create_distance_matrix(city_locations):
    import numpy as np
    distance = np.empty(shape=(len(city_locations), len(city_locations)))
    for i_city, i_cord in enumerate(city_locations):
        for j_city, j_cord in enumerate(city_locations):

            distance[i_city, j_city] = np.linalg.norm(j_cord - i_cord)

    return distance


def generate_random_punishment_graph(nbr_transport_types, nbr_cities):
    import numpy as np

    max_nbr_connections = nbr_transport_types * nbr_cities * nbr_cities
    lower_nbr_connection_limit = int(0.2 * max_nbr_connections)
    higher_nbr_connection_limit = int(0.5 * max_nbr_connections)
    nbr_connections = np.random.randint(lower_nbr_connection_limit, higher_nbr_connection_limit)

    max_punishment = 200
    min_punishment = 5

    punishment_graph = np.full(shape=(nbr_transport_types, nbr_cities, nbr_cities), fill_value=np.NAN)

    for i_connection in range(nbr_connections):

        transport_type = np.random.randint(nbr_transport_types)
        from_city = np.random.randint(nbr_cities)
        to_city = np.random.randint(nbr_cities)
        punishment = np.random.randint(min_punishment, max_punishment)

        punishment_graph[transport_type, from_city, to_city] = punishment

    return punishment_graph


def generate_punishment_graph_from_distance(nbr_transport_types, distance_matrix, extra_points):
    """
    score: 1/(travel time + punishment) where travel time is 'distance'/'transport_type_speed' and punishment is
    'transport_type_punishment'

    :param nbr_transport_types:
    :param distance_matrix:
    :return:
    """
    import numpy as np

    nbr_cities = len(distance_matrix)

    transport_type_speed = [20, 100, 90, 50, 1000]   # i km/h: cykel, bil, buss, båt, flyg
    transport_type_punishment = [1, 80, 55, 35, 100]    # per användning
    time_punishment_ratio = 0.8

    punishment_graph = np.full(shape=(nbr_transport_types, nbr_cities, nbr_cities), fill_value=np.NAN)
    travel_time_graph = np.full(shape=(nbr_transport_types, nbr_cities, nbr_cities), fill_value=np.NAN)
    score_graph = np.full(shape=(nbr_transport_types, nbr_cities, nbr_cities), fill_value=np.NAN)

    max_dist = np.max(distance_matrix)

    for from_city in range(nbr_cities):
        for to_city in range(nbr_cities):
            dist = distance_matrix[from_city, to_city]

            # Set transport depending on distance
            max_transport_type = min(int(dist/max_dist*10/2 + 0.5), 5)
            trans_types = list(range(max_transport_type))

            # Randomly connect some flights and boats due to low prob. above
            rnd = np.random.random()
            if rnd < 0.10:
                trans_types.append(4)
            if rnd < 0.20:
                trans_types.append(3)
            #if rnd < 0.40:
            #    trans_types.append(3)
            #if rnd < 0.60:
            #    trans_types.append(2)
            #if rnd < 0.80:
            #    trans_types.append(1)
            #trans_types.append(0)
            trans_types = set(trans_types)

            for k_transport_type in trans_types:

                # Calculate score and travel time
                travel_time = dist/transport_type_speed[k_transport_type]
                score = (1+extra_points[to_city])/(travel_time + time_punishment_ratio * transport_type_punishment[k_transport_type])

                # Add values to graphs
                travel_time_graph[k_transport_type, from_city, to_city] = travel_time

                punishment_graph[k_transport_type, from_city, to_city] = \
                    travel_time + time_punishment_ratio*transport_type_punishment[k_transport_type]

                score_graph[k_transport_type, from_city, to_city] = score

    return travel_time_graph, punishment_graph, score_graph


def generate_random_city_extra_points(nbr_cities, max_point=1):
    import numpy as np

    return np.random.rand(nbr_cities) * max_point
