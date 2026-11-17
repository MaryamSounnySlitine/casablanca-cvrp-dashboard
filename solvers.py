"""CVRP solvers for the Casablanca capstone.

The three methods are ported from the project's Colab notebook
(Casablanca_CVRP_Cleaned_Model) and only parameterised, so that with the
default inputs they follow the same logic as the thesis experiments:

* Nearest Neighbor: greedy nearest feasible customer.
* OR-Tools: Path Cheapest Arc + Guided Local Search, fixed vehicle cost
  (minimise vehicles first, then distance).
* Tabu Search: starts from the NN solution; relocate / swap /
  intra-route swap / route-elimination moves; lexicographic objective
  (vehicles, distance); aspiration criterion.

Every solver works on a sub-instance: node 0 is the depot, nodes 1..n are
the selected customers. All solvers return a list of routes, each route a
list of node indices WITHOUT the depot at either end.
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def route_distance(route, dist):
    full = [0] + list(route) + [0]
    return float(sum(dist[full[i], full[i + 1]] for i in range(len(full) - 1)))


def route_load(route, demands):
    return int(sum(demands[c] for c in route))


# --------------------------------------------------------------------------
# Nearest Neighbor
# --------------------------------------------------------------------------

def nearest_neighbor(dist, demands, capacity, max_vehicles=None):
    n = len(demands)
    unvisited = set(range(1, n))
    routes = []
    t0 = time.time()

    while unvisited:
        if max_vehicles is not None and len(routes) >= max_vehicles:
            break
        route, current, remaining = [], 0, capacity
        while True:
            feasible = [c for c in unvisited if demands[c] <= remaining]
            if not feasible:
                break
            nxt = min(feasible, key=lambda c: dist[current, c])
            route.append(nxt)
            unvisited.remove(nxt)
            remaining -= demands[nxt]
            current = nxt
        if not route:  # a customer bigger than the truck
            break
        routes.append(route)

    return {
        "routes": routes,
        "unserved": sorted(unvisited),
        "time_s": time.time() - t0,
        "history": None,
    }


# --------------------------------------------------------------------------
# OR-Tools Guided Local Search
# --------------------------------------------------------------------------

def ortools_gls(dist, demands, capacity, max_vehicles, time_limit_s=120,
                minimise_vehicles=True):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    t0 = time.time()
    dist_m = np.rint(dist * 1000).astype(int)
    n = len(demands)

    manager = pywrapcp.RoutingIndexManager(n, max_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(i, j):
        return int(dist_m[manager.IndexToNode(i)][manager.IndexToNode(j)])

    transit = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    if minimise_vehicles:
        max_arc = int(dist_m.max())
        routing.SetFixedCostOfAllVehicles(max_arc * (n + max_vehicles) + 1)

    def demand_cb(i):
        return int(demands[manager.IndexToNode(i)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [int(capacity)] * max_vehicles, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = int(time_limit_s)

    solution = routing.SolveWithParameters(params)
    routes = []
    if solution:
        for v in range(max_vehicles):
            idx = routing.Start(v)
            r = []
            while not routing.IsEnd(idx):
                idx = solution.Value(routing.NextVar(idx))
                if not routing.IsEnd(idx):
                    r.append(manager.IndexToNode(idx))
            if r:
                routes.append(r)

    served = {c for r in routes for c in r}
    return {
        "routes": routes,
        "unserved": sorted(set(range(1, n)) - served),
        "time_s": time.time() - t0,
        "history": None,
        "status": "ok" if solution else "no feasible solution found",
    }


# --------------------------------------------------------------------------
# Tabu Search
# --------------------------------------------------------------------------

def tabu_search(dist, demands, capacity, initial_routes, iterations=300,
                tabu_tenure=15, neighbors_per_iteration=20, seed=4394,
                progress=None):
    """Tabu Search as in the notebook (moves, tenure, aspiration)."""
    rng = random.Random(seed)
    dist_m = np.rint(dist * 1000).astype(int)
    n = len(demands)
    all_customers = set(range(1, n))

    def load(r):
        return sum(demands[c] for c in r)

    def rdist(r):
        full = [0] + r + [0]
        return sum(dist_m[full[i]][full[i + 1]] for i in range(len(full) - 1))

    def sdist(routes):
        return sum(rdist(r) for r in routes)

    def feasible(routes):
        assigned = [c for r in routes for c in r]
        ok = (len(assigned) == n - 1 and set(assigned) == all_customers)
        return ok and all(load(r) <= capacity for r in routes)

    def evaluate(routes):
        return (len(routes), sdist(routes))

    def neighbor(routes, max_attempts=100):
        for _ in range(max_attempts):
            new = copy.deepcopy(routes)
            mt = rng.choice(["relocate", "swap", "intra_route_swap",
                             "route_elimination"])

            if mt == "route_elimination":
                if len(new) <= 1:
                    continue
                src = new.pop(rng.randrange(len(new)))
                src.sort(key=lambda c: demands[c], reverse=True)
                ok = True
                for c in src:
                    options = [i for i, r in enumerate(new)
                               if load(r) + demands[c] <= capacity]
                    if not options:
                        ok = False
                        break
                    t = rng.choice(options)
                    new[t].insert(rng.randrange(len(new[t]) + 1), c)
                if not ok:
                    continue
                move = ("route_elimination", tuple(sorted(src)))

            elif mt == "relocate":
                if len(new) < 2:
                    continue
                s = rng.randrange(len(new))
                c = new[s].pop(rng.randrange(len(new[s])))
                t = rng.choice([i for i in range(len(new)) if i != s])
                new[t].insert(rng.randrange(len(new[t]) + 1), c)
                new = [r for r in new if r]
                move = ("relocate", c, t)

            elif mt == "swap":
                if len(new) < 2:
                    continue
                a, b = rng.sample(range(len(new)), 2)
                pa, pb = rng.randrange(len(new[a])), rng.randrange(len(new[b]))
                ca, cb = new[a][pa], new[b][pb]
                new[a][pa], new[b][pb] = cb, ca
                move = ("swap", min(ca, cb), max(ca, cb))

            else:
                ri = rng.randrange(len(new))
                if len(new[ri]) < 2:
                    continue
                pa, pb = rng.sample(range(len(new[ri])), 2)
                ca, cb = new[ri][pa], new[ri][pb]
                new[ri][pa], new[ri][pb] = cb, ca
                move = ("intra_route_swap", min(ca, cb), max(ca, cb))

            if feasible(new) and new != routes:
                return new, move
        return None, None

    t0 = time.time()
    current = [list(r) for r in initial_routes if r]
    best = copy.deepcopy(current)
    best_score = evaluate(best)
    tabu = {}
    history = []

    for it in range(iterations):
        bn, bm, bs = None, None, None
        for _ in range(neighbors_per_iteration):
            nb, mv = neighbor(current)
            if nb is None:
                continue
            sc = evaluate(nb)
            if tabu.get(mv, -1) > it and sc >= best_score:
                continue
            if bs is None or sc < bs:
                bn, bm, bs = nb, mv, sc
        if bn is not None:
            current = bn
            tabu[bm] = it + tabu_tenure
            if bs < best_score:
                best, best_score = copy.deepcopy(current), bs
            tabu = {m: e for m, e in tabu.items() if e > it}
        history.append({"Iteration": it + 1,
                        "Vehicles": best_score[0],
                        "Distance_km": round(best_score[1] / 1000, 2)})
        if progress is not None:
            progress((it + 1) / iterations)

    return {
        "routes": best,
        "unserved": [],
        "time_s": time.time() - t0,
        "history": history,
    }
