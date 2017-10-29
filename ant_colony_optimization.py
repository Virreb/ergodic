

def initiate_pheromones(score_graph):
    import numpy as np

    norm = np.sqrt(np.nansum(np.square(score_graph)))

    return score_graph / norm


def walk_with_ant(pheromones, start_city:int, goal_city:int, parameters:int) -> list:
    # TODO

    path = []

    return path


def evaluate_path(score_graph, city_extra_points, ant_path):
    # TODO

    score = 0.0

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
            score = evaluate_path()

            all_ant_paths.append(path)
            all_ant_scores.append(score)

        update_pheromones()
        score_std = np.std(all_ant_scores)

    ultimate_ant_travel_path = 'ultra fast, ultra miljövänlig'
    ultimate_travel_time = 0.5
    ultimate_punishment = 0.1

    return ultimate_ant_travel_path, ultimate_travel_time, ultimate_punishment
