"""
Ant Colony Optimisation (ACO) for Wind Turbine Vessel Routing
Finds the best order to visit N turbines from a port and return.
"""

import numpy as np
import random


class ACORoutePlanner:
    def __init__(
        self,
        n_ants: int = 40,
        n_iterations: int = 120,
        alpha: float = 1.2,    # pheromone importance
        beta: float = 2.5,     # heuristic importance
        evaporation: float = 0.4,
        q: float = 100.0,
    ):
        self.n_ants       = n_ants
        self.n_iterations = n_iterations
        self.alpha        = alpha
        self.beta         = beta
        self.evaporation  = evaporation
        self.q            = q

    def _build_cost_matrix(self, locations: list[dict]) -> np.ndarray:
        """locations: list of {"id", "lat", "lon", "pred_time"}"""
        n = len(locations)
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Haversine-based travel cost weighted by predicted time
                    lat1, lon1 = np.radians(locations[i]["lat"]), np.radians(locations[i]["lon"])
                    lat2, lon2 = np.radians(locations[j]["lat"]), np.radians(locations[j]["lon"])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
                    dist_km = 2 * 6371 * np.arcsin(np.sqrt(a))
                    # blend geographic distance with predicted travel time
                    avg_time = (locations[i]["pred_time"] + locations[j]["pred_time"]) / 2
                    cost[i][j] = 0.6 * dist_km + 0.4 * avg_time * 15   # scale time to km units
        return cost

    def optimise(self, locations: list[dict]) -> dict:
        """
        locations: list of {"id", "lat", "lon", "pred_time"}
        Returns: {"route": [...ids...], "total_cost": float, "convergence": [...]}
        """
        n = len(locations)
        if n <= 1:
            return {
                "route": [loc["id"] for loc in locations],
                "total_cost": sum(loc["pred_time"] for loc in locations),
                "convergence": []
            }

        cost = self._build_cost_matrix(locations)
        pheromone = np.ones((n, n))
        heuristic = 1.0 / (cost + 1e-10)

        best_route = None
        best_cost  = float("inf")
        convergence = []

        for iteration in range(self.n_iterations):
            all_routes = []
            all_costs  = []

            for _ in range(self.n_ants):
                route = self._build_route(n, pheromone, heuristic)
                route_cost = sum(cost[route[i]][route[i+1]] for i in range(len(route)-1))
                all_routes.append(route)
                all_costs.append(route_cost)

                if route_cost < best_cost:
                    best_cost  = route_cost
                    best_route = route[:]

            # Pheromone evaporation
            pheromone *= (1 - self.evaporation)

            # Pheromone deposit
            for route, rc in zip(all_routes, all_costs):
                deposit = self.q / (rc + 1e-10)
                for i in range(len(route) - 1):
                    pheromone[route[i]][route[i+1]] += deposit
                    pheromone[route[i+1]][route[i]] += deposit

            convergence.append(best_cost)

        route_ids = [locations[i]["id"] for i in best_route]

        # Calculate actual total travel time along optimised route
        total_time = sum(locations[i]["pred_time"] for i in best_route)

        return {
            "route": route_ids,
            "route_indices": best_route,
            "total_cost": best_cost,
            "total_time_hrs": round(total_time, 2),
            "convergence": convergence,
        }

    def _build_route(self, n: int, pheromone, heuristic) -> list:
        start = 0   # port is always index 0
        visited = [False] * n
        route   = [start]
        visited[start] = True

        current = start
        while len(route) < n:
            probs = []
            candidates = []
            for j in range(n):
                if not visited[j]:
                    p = (pheromone[current][j] ** self.alpha) * (heuristic[current][j] ** self.beta)
                    probs.append(p)
                    candidates.append(j)

            if not candidates:
                break

            total = sum(probs)
            if total == 0:
                next_node = random.choice(candidates)
            else:
                probs = [p / total for p in probs]
                next_node = random.choices(candidates, weights=probs, k=1)[0]

            route.append(next_node)
            visited[next_node] = True
            current = next_node

        route.append(start)   # return to port
        return route
