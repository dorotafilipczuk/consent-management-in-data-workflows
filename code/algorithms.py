#!/usr/bin/env python
"""Provides graph-cutting algorithms."""

from collections import deque
import itertools
import networkx as nx
import picos as pic
import random
import sys

__author__ = "Dorota Filipczuk"
__copyright__ = "Copyright 2020, University of Southampton"
__credits__ = "Dorota Filipczuk"
__license__ = "CC"
__version__ = "1.0.1"
__maintainer__ = "Dorota Filipczuk"
__email__ = "dorota@ecs.soton.ac.uk"
__status__ = "Prototype"


"""Given an edge which is about to be removed, decrease the values of its
dependencies or remove them completely if 0."""
def _update_dependences(graph, edge):
  importance = graph.get_edge_data(*edge)['capacity']
  dependencies = deque(graph.out_edges(edge[1]))
  while len(dependencies) > 0:
      dependency = dependencies.popleft()
      if graph.has_edge(*dependency):
          graph.get_edge_data(*dependency)['capacity'] -= importance
          for child_dependency in graph.out_edges(dependency[1]):
            dependencies.append(child_dependency)
          if graph.get_edge_data(*dependency)['capacity'] <= 0:
            graph.remove_edge(*dependency)

"""Algorithm 0: disconnect the source."""
def disconnect_the_source(graph, constraints):
  for constraint in constraints:
    for edge in list(graph.out_edges(constraint[0])):
      _update_dependences(graph, edge)
      graph.remove_edge(*edge)

"""Algorithm 1: remove the first edge from each path."""
def remove_first_edge(graph, constraints):
  for constraint in constraints:
    source, target = constraint
    edge_paths = list(nx.all_simple_edge_paths(graph, source=source,
          target=target))
    for edge_path in edge_paths:
      edge = edge_path[0]
      if graph.has_edge(*edge):
        _update_dependences(graph, edge)
        graph.remove_edge(*edge)

"""Algorithm 2: remove random edges."""
def remove_random_edge(graph, constraints):
  for constraint in constraints:
    print("constraint: " + str(constraint))
    source, target = constraint
    edge_paths = list(nx.all_simple_edge_paths(graph, source=source,
          target=target))
    for edge_path in edge_paths:
      edge = random.choice(edge_path)
      if graph.has_edge(*edge):
        _update_dependences(graph, edge)
        graph.remove_edge(*edge)

"""Algorithm 3: satisfy each constraint separately, one by one."""
def _get_edges(graph, nodes):
  edges = []
  for node in nodes:
    edges.extend(list(graph.in_edges(node, data=True)))
  return edges

def _get_number_of_paths(graph):
  nodes = set()
  for node in graph.nodes:
    if graph.out_degree(node) == 0:
      nodes.add(node)
  edges = _get_edges(graph, nodes)

  while len(edges) > 0:
    nodes = set()
    for edge in edges:
      graph.get_edge_data(*edge)['paths'] = graph.out_degree(edge[1], weight='paths') if graph.out_degree(edge[1], weight='paths') > 0 else 1
      nodes.add(edge[0])
    edges = _get_edges(graph, nodes)

def _assign_capacities(graph):
  nodes = set()
  for node in graph.nodes:
    if graph.out_degree(node) == 0:
      nodes.add(node)
  edges = _get_edges(graph, nodes)

  while len(edges) > 0:
    nodes = set()
    for edge in edges:
      graph.get_edge_data(*edge)['paths'] = graph.out_degree(edge[1], weight='paths') if graph.out_degree(edge[1], weight='paths') > 0 else 1
      graph.get_edge_data(*edge)['multiplied_capacity'] = graph.get_edge_data(*edge)['capacity'] * graph.get_edge_data(*edge)['paths']
      nodes.add(edge[0])
    edges = _get_edges(graph, nodes)

def remove_st_cuts(graph, constraints):
  for (source, target) in constraints:
    _assign_capacities(graph)
    cut_value, partition = nx.minimum_cut(graph, source, target, capacity='multiplied_capacity')
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, graph[n]) for n in reachable):
      cutset.update((u, v) for v in nbrs if v in non_reachable)
    for edge in cutset:
      if graph.has_edge(*edge):
        _update_dependences(graph, edge)
        graph.remove_edge(*edge)

"""Algorithm 4: based on the minimum multicut solver."""
"""The solution based on minimum multicut."""
def _get_minimum_multicut(G, pairs):
  # Number of nodes.
  N = G.number_of_nodes()

  # Generate edge capacities.
  c={}
  for e in sorted(G.edges(data=True)):
    capacity = G.get_edge_data(*e)['multiplied_capacity']
    e[2]['multiplied_capacity'] = capacity
    c[(e[0], e[1])]  = capacity

  # Convert the capacities to a PICOS expression.
  cc=pic.new_param('c',c)

  multicut = pic.Problem()

  # Extract the sources and sinks.
  sources=set([p[0] for p in pairs])
  sinks=set([p[1] for p in pairs])

  # Define the cut indicator variables.
  y={}
  for e in G.edges():
    y[e]=multicut.add_variable('y[{0}]'.format(e), vtype='binary')

  # Define one potential for each source.
  p={}
  for s in sources:
    p[s]=multicut.add_variable('p[{0}]'.format(s), N)

  # State the potential inequalities.
  multicut.add_list_of_constraints(
      [y[i,j] >= p[s][i]-p[s][j] for s in sources for (i,j) in G.edges()])

  # Set the source potentials to one.
  multicut.add_list_of_constraints([p[s][s] == 1 for s in sources])

  # Set the sink potentials to zero.
  multicut.add_list_of_constraints([p[s][t] == 0 for (s,t) in pairs])

  # Enforce nonnegativity.
  multicut.add_list_of_constraints([p[s] >= 0 for s in sources])

  # Set the objective.
  multicut.set_objective('min', pic.sum([cc[e]*y[e] for e in G.edges()]))

  # Solve the problem.
  multicut.solve(solver='glpk')

  # Extract the cut.
  return [e for e in G.edges() if round(y[e]) == 1]

def solver_based(graph, constraints):
  _assign_capacities(graph)
  multicut = _get_minimum_multicut(graph, constraints)
  for edge in multicut:
    if graph.has_edge(*edge):
      _update_dependences(graph, edge)
      graph.remove_edge(*edge)

"""Algorithm 5: iterative brute force."""
def _update_current_importances(graph, edge):
  importance = graph.get_edge_data(*edge)['capacity']
  graph.get_edge_data(*edge)['current_weight'] = 0
  dependencies = deque(graph.out_edges(edge[1]))
  while len(dependencies) > 0:
    dependency = dependencies.popleft()
    if graph.has_edge(*dependency):
      graph.get_edge_data(*dependency)['current_weight'] -= importance
      for child_dependency in graph.out_edges(dependency[1]):
        dependencies.append(child_dependency)

"""Zero all edges in @edges and all of their dependencies."""
def _to_be_zeroed(graph, edges):
  while len(edges) > 0:
    edge = edges.popleft()
    graph.get_edge_data(*edge)['current_weight'] = 0
    graph.get_edge_data(*edge)['current_paths'] = 0
    if graph.in_degree(edge[1], weight='current_paths') == 0:
      for dependency in graph.out_edges(edge[1], data=True):
        edges.append(dependency)

def _initialise(graph):
  nodes = set()
  for node in graph.nodes:
    if graph.out_degree(node) == 0:
      nodes.add(node)
  edges = _get_edges(graph, nodes)

  while len(edges) > 0:
    nodes = set()
    for edge in edges:
      graph.get_edge_data(*edge)['paths'] = graph.out_degree(edge[1], weight='paths') if graph.out_degree(edge[1], weight='paths') > 0 else 1
      graph.get_edge_data(*edge)['current_weight'] = graph.get_edge_data(*edge)['capacity']
      graph.get_edge_data(*edge)['current_paths'] = graph.get_edge_data(*edge)['paths']
      nodes.add(edge[0])
    edges = _get_edges(graph, nodes)

"""@edges - a list of the last edges from all paths considered.
@edges is of type `deque'!"""
def _update_current_paths(graph, edges):
  while len(edges) > 0:
    edge = edges.popleft()
    if graph.out_degree(edge[1]) == 0:
      if graph.get_edge_data(*edge)['current_weight'] == 0:
        graph.get_edge_data(*edge)['current_paths'] = 0
      for e in graph.in_edges(edge[0], data=True):
        edges.append(e)

    elif not (graph.get_edge_data(*edge)['current_weight'] == 0 and graph.get_edge_data(*edge)['current_paths'] == 0):
      if graph.get_edge_data(*edge)['current_weight'] == 0:
        graph.get_edge_data(*edge)['current_paths'] = 0
        if graph.in_degree(edge[1], weight='current_paths') == 0:
          _to_be_zeroed(graph, deque(graph.out_edges(edge[1], data=True)))

      graph.get_edge_data(*edge)['current_paths'] = graph.out_degree(edge[1], weight='current_paths')
      for e in graph.in_edges(edge[0], data=True):
        edges.append(e)

def matrix_based_bruteforce(graph, constraints):
  _initialise(graph)

  """Get all edge paths between the constraints."""
  all_edge_paths = []
  last_edges = []
  for source, target in constraints:
    edge_paths = list(nx.all_simple_edge_paths(graph, source=source, target=target))
    if len(edge_paths) > 0:
      all_edge_paths.extend(edge_paths)
  last_edges = deque([edge_path[-1] for edge_path in all_edge_paths])

  """Check all combinations iteratively."""
  indexes = [0] * len(all_edge_paths)
  min_loss = sys.maxsize
  min_multicut = set()
  current_loss = 0
  current_multicut = set()
  all_checked = False

  while not all_checked:
    for path, index in zip(all_edge_paths, indexes):
      edge = path[index]
      if edge not in current_multicut:
        current_multicut.add(edge)
        current_loss += graph.get_edge_data(*edge)['current_weight'] * graph.get_edge_data(*edge)['current_paths']
        _update_current_importances(graph, edge)
        _update_current_paths(graph, last_edges)

    if current_loss < min_loss:
      min_loss = current_loss
      min_multicut = current_multicut

    current_multicut = set()
    current_loss = 0
    _initialise(graph)

    j = len(indexes) - 1
    while True:
      indexes[j] += 1
      if indexes[j] < len(all_edge_paths[j]):
        break
      indexes[j] = 0
      j -= 1
      if j < 0:
        all_checked = True
        break

  for edge in min_multicut:
    if graph.has_edge(*edge):
      _update_dependences(graph, edge)
      graph.remove_edge(*edge)

"""Algorithm 6: old optimisation with pruning."""
def optimisation_approx(graph, constraints):
  """Get all edge paths between the constraints."""
  _get_number_of_paths(graph)
  all_edge_paths = []
  for source, target in constraints:
    edge_paths = list(nx.all_simple_edge_paths(graph, source=source,
        target=target))
    if len(edge_paths) > 0:
      all_edge_paths.extend(edge_paths)

  for edge_path in all_edge_paths:
    for edge in edge_path:
      graph.get_edge_data(*edge)['occurences'] = 0
      if 'multiplied_capacity' not in graph.get_edge_data(*edge):
        graph.get_edge_data(*edge)['multiplied_capacity'] = graph.get_edge_data(
            *edge)['capacity'] * graph.get_edge_data(*edge)['paths']

  """Check all combinations iteratively."""
  indices = [0] * len(all_edge_paths) # Indicate the next edge to take.
  min_loss = sys.maxsize
  min_multicut = set()
  current_loss = 0
  current_multicut = set()
  path_index = 0

  while path_index >= 0:
    while path_index < len(all_edge_paths):
      path = all_edge_paths[path_index]

      """Sort by the potential loss if edge removed."""
      if indices[path_index] == 0:
        path.sort(key=lambda e: graph.get_edge_data(*e)['multiplied_capacity']
            if graph.get_edge_data(*e)['occurences'] == 0 else 0)

      """Get an edge to add to the multicut. I can't, then backtrack."""
      if indices[path_index] == len(path):
        indices[path_index] = 0
        path_index -= 1
        """Remove the previous edge."""
        edge = path[indices[path_index] - 1]
        graph.get_edge_data(*edge)['occurences'] -= 1
        if graph.get_edge_data(*edge)['occurences'] == 0:
          current_multicut.remove(edge)
          current_loss -= graph.get_edge_data(*edge)['multiplied_capacity']
        if path_index < 0:
          break

      else:
        edge = path[indices[path_index]]
        indices[path_index] += 1
        edge_loss = graph.get_edge_data(*edge)['multiplied_capacity']
        if graph.get_edge_data(*edge)['occurences'] != 0:
          edge_loss = 0

        if current_loss + edge_loss >= min_loss:
          """Backtrack."""
          indices[path_index] = 0
          path_index -= 1
          """Remove the previous edge."""
          path = all_edge_paths[path_index]
          edge = path[indices[path_index] - 1]
          graph.get_edge_data(*edge)['occurences'] -= 1
          if graph.get_edge_data(*edge)['occurences'] == 0:
            current_multicut.remove(edge)
            current_loss -= graph.get_edge_data(*edge)['multiplied_capacity']
          if path_index < 0:
            break
        else:
          """Edge found. Add it to the current multicut."""
          current_multicut.add(edge)
          graph.get_edge_data(*edge)['occurences'] += 1
          current_loss += edge_loss
          path_index += 1

    if current_loss == 0:
      break

    """Multicut complete. Is it minimal?"""
    """If current_loss == min_loss, then add the solution to the equivalent
       solutions!"""
    if current_loss < min_loss:
      min_multicut = current_multicut.copy()
      min_loss = current_loss

    """Check the alternatives."""
    path_index -= 1
    path = all_edge_paths[path_index]
    edge = path[indices[path_index] - 1]
    graph.get_edge_data(*edge)['occurences'] -= 1
    if graph.get_edge_data(*edge)['occurences'] == 0:
      current_multicut.remove(edge)
      current_loss -= graph.get_edge_data(*edge)['multiplied_capacity']

  """Finalise."""
  for edge in min_multicut:
    if graph.has_edge(*edge):
      _update_dependences(graph, edge)
      graph.remove_edge(*edge)

  #return min_loss
"""A7"""
"""Calculate the exact utility of a graph."""
def get_utility(graph, graph_size):
    graph_utility = 0
    start_node = int(graph_size - (graph_size * 0.05))
    for sink in range(start_node, graph_size):
        if sink in graph.nodes:
            for edge in graph.in_edges(sink, data=True):
                graph_utility += graph.get_edge_data(*edge)['capacity']
    return graph_utility

def bruteforce(graph, constraints):
  """Get all edge paths between the constraints."""
  graph_size = graph.number_of_nodes()
  all_edge_paths = []
  last_edges = []
  for source, target in constraints:
    edge_paths = list(nx.all_simple_edge_paths(graph, source=source, target=target))
    if len(edge_paths) > 0:
      all_edge_paths.extend(edge_paths)
  last_edges = deque([edge_path[-1] for edge_path in all_edge_paths])

  """Check all combinations iteratively."""
  best_solution = None
  best_utility = 0
  combinations = itertools.product(*all_edge_paths)
  for multicut in combinations:
    solution = graph.copy()
    for edge in multicut:
      if solution.has_edge(*edge):
        _update_dependences(solution, edge)
        solution.remove_edge(*edge)
    utility = get_utility(solution, graph_size)

    if best_solution == None or utility > best_utility:
      best_solution = solution
      best_utility = utility
      print(best_utility)

  return best_solution
