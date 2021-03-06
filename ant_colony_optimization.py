import numpy as np


class AntGotLostException(Exception):
    pass


def initiate_pheromones(score_graph):
    norm = np.sqrt(np.nansum(np.square(score_graph)))

    return score_graph/norm


def simplify_graph_to_matrix(punishment_graph):
    punishment_matrix = np.nanmin(punishment_graph, axis=0)
    transport_matrix = np.nanargmin(punishment_graph, axis=0)
    return punishment_matrix, transport_matrix


def get_next_city(current_city, city_extra_points, pheromone_levels, punishment_graph, alpha, beta):

    available_pheromone_levels = pheromone_levels[current_city, :]
    punishment_vector = punishment_graph[current_city, :]
    score_matrix = (1+city_extra_points) / punishment_vector

    probability_vector = available_pheromone_levels**alpha * score_matrix**beta
    probability_vector = probability_vector / np.nansum(probability_vector)  # normalize

    cumulative_prob_vector = np.nancumsum(probability_vector)

    r = np.random.rand()
    next_city = np.searchsorted(cumulative_prob_vector, r)
    return next_city


def get_ant_path(city_extra_points, punishment_matrix, transport_matrix, start_city, target_city, pheromone_levels, alpha, beta):

    current_city = start_city
    nbr_of_cities = len(city_extra_points)
    temp_city_extra_points = np.copy(city_extra_points)
    travelled_matrix = np.zeros(shape=punishment_matrix.shape)
    travelled_path = []    # list of tuples (transport_choice, from_node, to_node)

    target_node_reached = False
    i = 0
    while not target_node_reached:
        temp_city_extra_points[current_city] = 0  # no additional points for going to the same node again
        next_city = get_next_city(current_city, temp_city_extra_points, pheromone_levels, punishment_matrix, alpha, beta)
        travelled_matrix[current_city, next_city] += 1
        transport_choice = int(transport_matrix[current_city, next_city])
        travelled_path.append((transport_choice, current_city, next_city))
        current_city = next_city
        if next_city == target_city:
            target_node_reached = True
        i += 1
        #if i > nbr_of_cities**2:     # TODO: CHECK THIS PARAM
        #    raise AntGotLostException()

    return travelled_matrix, travelled_path


def evaluate_path(punishment_matrix, city_extra_points, travelled_matrix):
    total_punishment = np.nansum(punishment_matrix * travelled_matrix)
    visited_cities = np.sum(travelled_matrix > 0, axis=1) > 0
    total_city_extra_point = np.sum(city_extra_points[visited_cities])

    score = total_city_extra_point / total_punishment

    return score    # Maybe return travel time, punishment, score as different values


def update_pheromones(old_pheromones, all_paths, all_scores, evaporation):

    delta_pheromones = np.zeros(shape=old_pheromones.shape)
    for path, score in zip(all_paths, all_scores):
        delta_pheromones += path * score

    new_pheromones = (1 - evaporation) * old_pheromones + delta_pheromones

    return new_pheromones


def summon_the_ergodic_colony(punishment_matrix, transport_matrix, city_extra_points, start_city=0, target_city=1,
                              nbr_ants=30, nbr_max_iterations=500, nbr_min_iterations=10, evaporation=0.5, alpha=1.0,
                              beta=3.0, verbose=False, *args, **kwargs):
    import numpy as np
    import time

    pheromones = initiate_pheromones(punishment_matrix)

    std_treshold = 0.1    # should be a parameter
    score_std = std_treshold
    i_iteration = 0
    best_path = []
    best_score = 0
    start_time = time.time()
    nbr_lost_ants = 0
    while (score_std >= std_treshold or i_iteration < nbr_min_iterations) and i_iteration <= nbr_max_iterations:
        all_travelled_paths = list()
        all_scores = list()

        for ant in range(nbr_ants):
            try:
                travelled_matrix, path = get_ant_path(city_extra_points, punishment_matrix, transport_matrix, start_city, target_city, pheromones, alpha, beta)
                score = evaluate_path(punishment_matrix, city_extra_points, travelled_matrix)
                all_travelled_paths.append(travelled_matrix)
                all_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_path = path

            except AntGotLostException:
                nbr_lost_ants += 1

            if nbr_lost_ants > 0.5*nbr_ants:

                if verbose:
                    print('Too many ants were lost, aborting mission :(')

                return [], 0

        i_iteration += 1

        pheromones = update_pheromones(pheromones, all_travelled_paths, all_scores, evaporation)
        score_std = np.std(all_scores)

    computation_time = time.time() - start_time

    if verbose:
        print(f'\nThe Ergodic colony has converged in {i_iteration} of {nbr_max_iterations} iterations!\n'
              f'Best path: {best_path}\n'
              f'Best score: {best_score}\n'
              f'Computation time: {computation_time}')

    return best_path, best_score


def run_parallel_colonies(nbr_parallel_jobs, nbr_colonies, args, kwargs):
    from joblib import Parallel, delayed
    import pickle

    best_path = []
    best_score = 0

    for i in range(int(nbr_colonies/nbr_parallel_jobs)):
        try:
            result_batch = Parallel(n_jobs=nbr_parallel_jobs)(
                delayed(summon_the_ergodic_colony)(*args, **kwargs) for _ in range(nbr_parallel_jobs)
                )
        except KeyboardInterrupt:
            pass

        for res in result_batch:
            path = res[0]
            score = res[1]

            if score > best_score:
                best_score = score
                best_path = path

                with open('best.pkl', 'wb') as f:
                    pickle.dump((best_path, best_score), f)

    print(f'\n\nFinished! {nbr_colonies} colonies has converged.\n'
          f'Best score: {best_score}\nBest path: {best_path}')

    return best_path, best_score, result_batch