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


def add_random_paths_to_static_graph(distance_matrix, static_connection_graph):
    import numpy as np

    # transport_type_speed = [20, 100, 70, 30, 1000]   # i km/h: cykel, bil, buss, båt, flyg
    # start_time_offset = [0, 0, 1, 8, 24]    # per use
    # enviromental_offset = [0, 10, 3, 1, 60]     # per use
    # transport_type_punishment = [0, 10, 2, 1, 100]    # per km

    k = np.array([1.5, 0.3, 0.5, 0.7, 0.1]) / 10;
    m = [0, 3, 1.1, 1, 10]

    time_punishment_ratio = 0.8

    punishment_graph = np.copy(static_connection_graph)
    for transport_type, both_city_matrix in enumerate(punishment_graph):
        for from_city, to_city_vector in enumerate(both_city_matrix):
            for to_city, val in enumerate(to_city_vector):

                if (from_city is not to_city and np.isnan(val) and np.random.rand() < 0.001) or val == 1:
                    dist = distance_matrix[from_city, to_city] * (1 + 0.2*np.random.rand())     # roads are not fågelvägen mostly
                    # speed = transport_type_speed[transport_type]
                    # time_offset = start_time_offset[transport_type]
                    # env_offset = enviromental_offset[transport_type]
                    # env_cost = transport_type_punishment[transport_type]

                    # punishment = dist/speed + time_offset + time_punishment_ratio * (dist * env_cost + env_offset)
                    punishment = k[transport_type]*dist + m[transport_type]
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

            if rnd < 0.5 and 0 in trans_types:
                trans_types.remove(1)
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


def merge_graph_to_matrix(graph):
    nbr_of_cities = graph.shape[1]
    transport_matrix = np.zeros(shape=(nbr_of_cities, nbr_of_cities))
    matrix = np.zeros(shape=(nbr_of_cities, nbr_of_cities))
    for i in range(nbr_of_cities):
        for j in range(nbr_of_cities):
            transport_array = graph[:, i, j]
            try:
                transport_choice = np.nanargmin(transport_array)
                graph_value = graph[transport_choice, i, j]
                transport_matrix[i, j] = transport_choice
                matrix[i, j] = graph_value
            except ValueError:
                transport_matrix[i, j] = np.nan
                matrix[i, j] = np.nan

    return matrix, transport_matrix


def plot_graph(city_locations, punishment_graph, travelled_path):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.MultiGraph()

    pos = dict()
    labels = dict()
    # Add nodes and labels
    for city, loc in enumerate(city_locations):
        labels[city] = city
        pos[city] = (int(loc[0]), int(loc[1]))

    black_edges = []
    red_edges = []

    # Add edges with colors
    for transport_type, both_city_matrix in enumerate(punishment_graph):
        for from_city, to_city_vector in enumerate(both_city_matrix):
            for to_city, val in enumerate(to_city_vector):
                if not np.isnan(val):
                    if (transport_type, from_city, to_city) in travelled_path:
                        red_edges.append((from_city, to_city))
                    else:
                        black_edges.append((from_city, to_city))

    # Plot the graph
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color=0.25, node_size=500000)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, font_color='blue')
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
    plt.show()
