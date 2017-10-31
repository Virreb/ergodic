import numpy as np

def initiate_pheromones(score_graph):
    import numpy as np

    norm = np.sqrt(np.nansum(np.square(score_graph)))

    return score_graph / norm


def get_next_city(current_city, city_extra_points, phermone_levels, punishment_graph, alpha, beta):

    available_phermone_levels = phermone_levels[:, current_city, :]
    punishment_matrix = punishment_graph[:, current_city, :]
    score_matrix = (1+city_extra_points) / punishment_matrix

    probability_matrix = available_phermone_levels**alpha * score_matrix**beta
    probability_matrix = probability_matrix / probability_matrix.sum()  # normalize

    cummulative_prob_vector = probability_matrix.cumsum()

    r = np.random.rand()
    winning_vector_index = np.searchsorted(cummulative_prob_vector, r)
    tranpsort_choice, next_city = np.unravel_index(winning_vector_index, probability_matrix.shape)

    return tranpsort_choice, next_city


def get_ant_path(city_extra_points, punishment_graph, start_city, target_city, phermone_levels, alpha, beta):

    current_city = start_city
    temp_city_extra_points = np.copy(city_extra_points)
    travelled_path = np.zeros(shape=punishment_graph.shape)

    target_node_reached = False
    i = 0
    while not target_node_reached and i < 1000000:
        temp_city_extra_points[current_city] = 0  # no addidtional points for going to the same node again
        transport_choice, next_city = get_next_city(current_city, temp_city_extra_points, phermone_levels, punishment_graph, alpha, beta)
        travelled_path[transport_choice, current_city, next_city] += 1
        current_city = next_city
        if next_city == target_city:
            target_node_reached = True
        i += 1

    return travelled_path


def evaluate_path(punishment_graph, city_extra_points, travelled_path, start_city):
    # TODO
    total_punishment = np.nansum(punishment_graph * travelled_path)
    visited_cities = np.sum(travelled_path > 0, axis=(0, 1))
    total_city_extra_point = np.sum(city_extra_points[visited_cities])

    score = total_city_extra_point / total_punishment

    return score    # Maybe return travel time, punishment, score as different values


def update_pheromones(old_pheromones, all_ant_paths, parameters):
    # TODO
    new_pheromone = []

    return new_pheromone


def summon_the_ergodic_colony(graph, nbr_ants=30, nbr_max_iterations=500, start_city=0, goal_city=1, parameters=0,
                              *args, **kwargs):
    import numpy as np
    # TODO

    pheromones = initiate_pheromones(graph)

    std_treshold = 1    # should be a parameter
    score_std = std_treshold
    i_iteration = 0
    while score_std >= std_treshold and i_iteration <= nbr_max_iterations:
        all_ant_paths = list()
        all_ant_scores = list()

        for ant in range(nbr_ants):

            path = walk_with_ant(pheromones)
            score = evaluate_path(path)

            all_ant_paths.append(path)
            all_ant_scores.append(score)

        update_pheromones()
        score_std = np.std(all_ant_scores)

    ultimate_ant_travel_path = 'ultra fast, ultra miljövänlig'
    ultimate_travel_time = 0.5
    ultimate_punishment = 0.1

    return ultimate_ant_travel_path, ultimate_travel_time, ultimate_punishment
