from api import API
import numpy as np
import graph, ant_colony_optimization as aco
API_KEY = '9b2a9819-b49b-41c0-8692-149ea06dfe58'
_api = API(API_KEY)


def solve(game):

    # --- Available commands ---
    # TRAVEL [NORTH|SOUTH|WEST|EAST]
    # [BUS|TRAIN|FLIGHT] {CityName}
    # SET_PRIMARY_TRANSPORTATION [CAR|BIKE]

    # TODO: Implement your solution

    # Set constants
    #nbr_cities = 10
    #nbr_transport_types = 5     # 0: bicycle, 1: car, 2:buss, 3: boat, 4: plane

    # Parallel colonies specifics
    nbr_colonies = 8
    nbr_parallel_jobs = 4

    # Get gamestate variables
    map_start_city = (game.start.x, game.start.y)
    map_target_city = (game.end.x, game.end.y)
    time_limit = game.timeLimit
    pollutions_point_rate = game.pollutionsPointRate
    all_cities = game.cities

    # Fix transport data to lists
    trans_str_list = ['Bike', 'Car', 'Bus', 'Train', 'Boat', 'Flight']
    trans = game.transportation

    trans_pollution_per_hour, trans_speed, trans_travel_interval = [], [], []
    for name in trans_str_list:
        for d in trans:
            if d['name'] == name:
                trans_pollution_per_hour.append(d['pollutions'])
                trans_speed.append(d['speed'])

                if d['travelInterval'] is None:
                    trans_travel_interval.append(np.NAN)
                else:
                    trans_travel_interval.append(d['travelInterval'])

    # Fix maps
    punishment_graph = graph.generate_1d_vector_from_2d_map(game.map,
                                                            trans_pollution_per_hour,
                                                            trans_speed,
                                                            trans_pollution_per_hour)

    punishment_matrix, transport_matrix = graph.merge_graph_to_matrix(punishment_graph)
    print(punishment_graph)
    # Set arguments to ACO
    args = (punishment_matrix, transport_matrix, city_extra_points)
    kwargs = {
        'start_city': start_city,
        'target_city': target_city,
        'nbr_ants': 50,
        'verbose': True,
        'evaporation': 0.5,
        'alpha': 1.0,   # pheromones
        'beta': 3.0     # scores
    }

    # Run in parallel
    best_path, best_score, all_results = aco.run_parallel_colonies(nbr_parallel_jobs, nbr_colonies, args, kwargs)

    #    # Example solution
    #    solution = list()
    #    x = game.start.x
    #    y = game.start.y
    #    while x < game.end.x:
    #        x += 1
    #        solution.append("TRAVEL EAST")
    #    while y < game.end.y:
    #        y += 1
    #        solution.append("TRAVEL SOUTH")

    return solution


def main():
    _api.initGame()
    game = _api.getMyLastGame()
    print(game)

    #Or get by gameId:
    #game = _api.getGame()
    #solution = solve(game)
    #_api.submitSolution(solution, game.id)

main()






