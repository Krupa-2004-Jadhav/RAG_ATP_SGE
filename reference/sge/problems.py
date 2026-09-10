import numpy as np
from scipy.spatial import distance_matrix
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

INSTANCES = {
    "easy": {
        "num_cities": 5,
        "max_coord": 100,
        "num_vehicles": 1,
        "seeds": list(range(10)),
        "description": "5-city VRP, 10 random instances"
    },
    "medium": {
        "num_cities": 8,
        "max_coord": 100,
        "num_vehicles": 1,
        "seeds": list(range(10)),
        "description": "8-city VRP, 10 random instances"
    },
    "hard": {
        "num_cities": 10,
        "max_coord": 100,
        "num_vehicles": 1,
        "seeds": list(range(10)),
        "description": "10-city VRP, 10 random instances"
    }
}


class VRPProblem:
    def __init__(self, num_cities=5, max_coord=100, num_vehicles=1, seed=None):
        self.num_cities = num_cities
        self.max_coord = max_coord
        self.num_vehicles = num_vehicles
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.coords = np.random.randint(0, max_coord, size=(num_cities, 2))
        dm = distance_matrix(self.coords, self.coords)
        self.dist_mat = [list(dm[i].astype(int)) for i in range(num_cities)]

        self._optimal_cost = None
        self._optimal_route = None

    def compute_cost(self, route):
        if not route or len(route) < 2:
            return 99999
        interior = [c for c in route if c != 0]
        expected = set(range(1, self.num_cities))
        if set(interior) != expected:
            return 99999
        cost = 0
        for i in range(1, len(route)):
            a, b = route[i-1], route[i]
            if a >= self.num_cities or b >= self.num_cities:
                return 99999
            cost += self.dist_mat[a][b]
        return cost

    def solve_optimal(self):
        if self._optimal_cost is not None:
            return self._optimal_route, self._optimal_cost

        data = {
            "distance_matrix": self.dist_mat,
            "num_vehicles": self.num_vehicles,
            "depot": 0
        }
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]),
            data["num_vehicles"],
            data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(
            distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
            cost = self.compute_cost(route)
            self._optimal_route = route
            self._optimal_cost = cost
            return route, cost

        return None, None

    def create_prompt(self):
        lines = [
            "You are solving a Vehicle Routing Problem.",
            f"There are {self.num_cities} locations indexed 0 to {self.num_cities - 1}.",
            "Location 0 is the depot. The vehicle starts and ends at depot.",
            "",
            "Coordinates (index: x, y):"
        ]
        for i, (x, y) in enumerate(self.coords):
            lines.append(f"  {i}: ({x}, {y})")

        lines.append("")
        lines.append("Distance matrix:")
        lines.append(
            "     " + "  ".join(f"{j:4d}" for j in range(self.num_cities)))
        for i in range(self.num_cities):
            row = "  ".join(f"{self.dist_mat[i][j]:4d}"
                            for j in range(self.num_cities))
            lines.append(f"  {i}: {row}")

        lines.append("")
        lines.append("Goal: shortest route starting and ending at depot (0),")
        lines.append("visiting every other location exactly once.")
        lines.append("Return ONLY the route as a list e.g. [0, 3, 1, 2, 4, 0]")

        return "\n".join(lines)

    def parse_solution(self, text):
        import re
        import ast

        matches = re.findall(r'\[[\d,\s]+\]', text)
        for match in matches:
            try:
                route = ast.literal_eval(match)
                if not isinstance(route, list):
                    continue
                route = [int(x) for x in route]

                # detect and fix 1-indexed
                if route and max(route) >= self.num_cities:
                    route = [max(0, x-1) for x in route]

                route = [x for x in route if 0 <= x < self.num_cities]

                if route and route[0] != 0:
                    route = [0] + route
                if route and route[-1] != 0:
                    route = route + [0]

                cost = self.compute_cost(route)
                if cost < 99999:
                    return route

            except Exception:
                continue

        return [0] + list(range(1, self.num_cities)) + [0]

    def compute_gap(self, route):
        _, optimal_cost = self.solve_optimal()
        if optimal_cost is None or optimal_cost <= 0:
            return -1
        model_cost = self.compute_cost(route)
        return max(0, (model_cost - optimal_cost) / optimal_cost * 100)


def create_instance(difficulty, seed):
    config = INSTANCES[difficulty]
    return VRPProblem(
        num_cities=config["num_cities"],
        max_coord=config["max_coord"],
        num_vehicles=config["num_vehicles"],
        seed=seed
    )
