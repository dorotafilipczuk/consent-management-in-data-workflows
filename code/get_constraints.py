#!/usr/bin/env python
"""Provides functions to generate constraints.
"""

import algorithms as algo
import csv
import json
import networkx as nx
import random
import sys
import time

__author__ = "Dorota Filipczuk"
__copyright__ = "Copyright 2020, University of Southampton"
__credits__ = "Dorota Filipczuk"
__license__ = "CC"
__version__ = "1.0.1"
__maintainer__ = "Dorota Filipczuk"
__email__ = "dorota@ecs.soton.ac.uk"
__status__ = "Prototype"


def read_graph_from_file(graph_size, iteration):
    filename = "graphs/graph_" + graph_size + "_" + iteration + ".json"
    return nx.read_edgelist(filename, create_using=nx.DiGraph(), nodetype=int)

def read_constraints_from_file(graph_size, iteration):
    filename = "constraints/cons_" + graph_size + "_" + iteration + ".json"
    with open(filename, 'r', encoding='utf-8') as file:
         return json.load(file)

"""Calculate the exact utility of a graph."""
def get_utility(graph, graph_size):
    graph_utility = 0
    start_node = int(graph_size - (graph_size * 0.05))
    for sink in range(start_node, graph_size):
        if sink in graph.nodes:
            for edge in graph.in_edges(sink, data=True):
                graph_utility += graph.get_edge_data(*edge)['capacity']
    return graph_utility

def save_graph_to_file(graph_size, graph, iteration):
    filename = "a5_graphs/graph_" + graph_size + "_" + iteration + ".json"
    nx.write_edgelist(graph, filename)

def _distribute_vertices(graph_size, vertex_distribution):
    if graph_size < 100:
        raise ValueError("Error distributing vertices: graph size smaller than \
            100. Increase the graph size!\n")

    layers = [0] * len(vertex_distribution)
    for i in range(len(vertex_distribution)):
        layers[i] = int(vertex_distribution[i] * graph_size)

    return layers

"""Check if it is possible to find any pair of constraints in the graph."""
def _are_constraints_possible(graph, constraints, first_layer_start, first_layer_end, last_layer_start, last_layer_end):
    for s in range(first_layer_start, first_layer_end):
        for t in range(last_layer_start, last_layer_end):
            if (s, t) not in constraints and nx.has_path(graph, s, t):
                return True
    return False


"""Returns an array of pairs of vertices currently connected."""
def get_random_constraints(pairs, graph_size, graph, vertex_distribution):
    constraints = set()

    """Calculate the number of nodes on the first and last layer."""
    layers = _distribute_vertices(graph_size, vertex_distribution)
    first_nodes = layers[0]
    last_nodes = layers[len(layers) - 1]

    if first_nodes + last_nodes > graph_size:
        raise ValueError("Number of first and last nodes greater than the \
            graph size.\n")

    first_layer_start = 0
    first_layer_end = first_nodes - 1
    last_layer_start = graph_size - last_nodes
    last_layer_end = graph_size - 1

    for _ in range(pairs):

        """If no paths are found, end the search."""
        while True:
            s = random.randint(first_layer_start, first_layer_end)
            t = random.randint(last_layer_start, last_layer_end)

            """Only add constraints if selected nodes are not in the constraints
            set already and they are connected."""
            if (s, t) not in constraints and nx.has_path(graph, s, t):
                constraints.add((s, t))
                break

            if not _are_constraints_possible(graph, constraints, first_layer_start, first_layer_end, last_layer_start, last_layer_end):
                raise ValueError("Tried everything and there are no other ways to construct constraints! Probably not enough paths in the graph.\n")

            if first_nodes * last_nodes == len(constraints):
                break

    return constraints

"""Arguments: graph size."""
def main(argv):
    graph_size = argv[0]
    number_of_constraints = 50
    vertex_distribution = [0.5, 0.25, 0.1, 0.1, 0.05]

    for iteration in range(1, 31):
        random.seed(int(graph_size) + iteration)
        graph = read_graph_from_file(graph_size, str(iteration))
        constraints = get_random_constraints(number_of_constraints, int(graph_size), graph, vertex_distribution)
        with open("constraints/50_cons_" + str(graph_size) + "_" + str(iteration) + ".json", "w") as myfile:
            json.dump(list(constraints), myfile)

if __name__ == "__main__":
   main(sys.argv[1:])
