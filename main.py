from api import API
from sys import getsizeof
import numpy as np, scipy.sparse
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

    start_city = map_start_city[1]*1000 + map_start_city[0]
    target_city = map_target_city[1]*1000 + map_target_city[0]

    trans_pollution_per_hour, trans_speed, trans_travel_interval = graph.transport_dict_to_vector(game.transportation)

    # Fix maps
    punishment_graph = graph.generate_1d_vector_from_2d_map(game.map,
                                                            trans_pollution_per_hour,
                                                            trans_speed,
                                                            pollutions_point_rate)

    punishment_graph = graph.add_special_connections_to_punishment_graph(punishment_graph, game.cities,
                                                                         trans_pollution_per_hour, trans_speed,
                                                                         pollutions_point_rate)


    print(punishment_graph)

    # Set arguments to ACO
#    args = (punishment_matrix, transport_matrix)
#    kwargs = {
#        'start_city': start_city,
#        'target_city': target_city,
#        'nbr_ants': 50,
#        'verbose': True,
#        'evaporation': 0.5,
#        'alpha': 1.0,   # pheromones
#        'beta': 3.0     # scores
#    }

    # Run in parallel
    aco.summon_the_ergodic_colony(punishment_graph, start_city=start_city, target_city=target_city, nbr_ants=100, verbose=True)
    #best_path, best_score, all_results = aco.run_parallel_colonies(nbr_parallel_jobs, nbr_colonies, args, kwargs)

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
    solution = []
    return solution


def main():
    import pickle
    _api.initGame()
    game = _api.getMyLastGame()
    #print(game)
    #solution = solve(game)

    attr_list = ['id',  'timeLimit', 'pollutionsPointRate', 'transportation', 'cities', 'objectives', 'map']

    d = dict()
    for attr in attr_list:
        d[attr] = getattr(game, attr)
        print(type(d[attr]))

    with open('game.pkl', 'wb') as f:
        pickle.dump(d, f)


    #Or get by gameId:
    #game = _api.getGame()
    #solution = solve(game)
    #_api.submitSolution(solution, game.id)
    scipy.sparse

    a = np.random.rand(10, 10)
    b = scipy.sparse.coo_matrix(a)
    print(getsizeof(a), 'bytes')
    print(getsizeof(b), 'bytes')

    print(a)
    print(b)
main()






