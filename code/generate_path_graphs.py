#!/usr/bin/env python
"""Provides functions to generate consent graphs.

generate_graph is the function responsible for generating a connected,
directed acyclic n-vertex graph. get_random_constraints is the function
selecting constraints, which are pseudo-random pairs of vertices currently
connected.
"""

import json
import csv
import networkx as nx
import os
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

"""Returns the number of vertices on each layer of the graph.
@vertex_distribution must be expressed as an array of percentages. The elements
of the array must sum up to 1.
@graph_size must be greater or equal to 100.
"""
def _distribute_vertices(graph_size, vertex_distribution):
    if graph_size < 100:
        raise ValueError("Error distributing vertices: graph size smaller than \
            100. Increase the graph size!\n")

    layers = [0] * len(vertex_distribution)
    for i in range(len(vertex_distribution)):
        layers[i] = 50

    return layers

"""Generates edges between two given layers of vertices, provided that the
vertices are already in the graph.
@edge_coverage must be expressed as a percentage (a floating point number
between 0 and 1).
"""
def _generate_edges(graph,
                   edge_coverage,
                   starting_layer1_node,
                   starting_layer2_node,
                   number_of_layer1_nodes,
                   number_of_layer2_nodes):
    if not graph.has_node(starting_layer1_node):
        raise ValueError("Error generating edges: starting layer1 node not in \
            the graph.\n")
    if not graph.has_node(starting_layer2_node):
        raise ValueError("Error generating edges: starting layer2 node not in \
            the graph.\n")
    if edge_coverage > 1:
        raise ValueError("Error generating edges: edge coverage not a \
            percentage value. It should be a floating point variable less than 1.\n")

    number_of_edges = int(edge_coverage * number_of_layer1_nodes *
        number_of_layer2_nodes)
    while number_of_edges != 0:
        s_node = random.randint(starting_layer1_node,
                         starting_layer1_node + number_of_layer1_nodes - 1)
        t_node = random.randint(starting_layer2_node,
                         starting_layer2_node + number_of_layer2_nodes - 1)
        if graph.has_edge(s_node, t_node) == False:
            graph.add_edge(s_node, t_node)
            number_of_edges -= 1

def _is_outgoing_connectivity(graph, node):
    if not graph.has_node(node):
        raise ValueError("Error checking out-connectivity: node not in the \
            graph.\n")

    for (s, t) in graph.edges:
        if s == node:
            return True
    return False

def _is_incoming_connectivity(graph, node):
    if not graph.has_node(node):
        raise ValueError("Error checking out-connectivity: node not in the \
            graph.\n")

    for (s, t) in graph.edges:
        if t == node:
            return True
    return False

"""Connect a disconnected node to any other."""
def _connect_to_anything(graph, node, starting_layer2_node,
                        number_of_layer2_nodes):
    if not graph.has_node(node):
        raise ValueError("Error connecting the node to others: node not in the \
            graph.\n")
    if not graph.has_node(starting_layer2_node):
        raise ValueError("Error connecting the node to others: starting layer2 \
            node not in the graph.\n")
    if number_of_layer2_nodes < 1:
        raise ValueError("Error connecting the node to others: number of \
            layer2 nodes is less than 1.\n")
    if len(graph.nodes) < 2:
        raise ValueError("Error connecting the node to others: there are no \
            other nodes in the graph.\n")

    t_node = random.randint(starting_layer2_node,
                     starting_layer2_node + number_of_layer2_nodes - 1)
    graph.add_edge(node, t_node)

"""Connect any node to the disconnected node."""
def _connect_from_anything(graph, node, starting_layer1_node,
                          number_of_layer1_nodes):
    if not graph.has_node(node):
        raise ValueError("Error connecting the node from others: node not in \
            the graph.\n")
    if not graph.has_node(starting_layer1_node):
        raise ValueError("Error connecting the node to others: starting layer1 \
            node not in the graph.\n")
    if number_of_layer1_nodes < 1:
        raise ValueError("Error connecting the node from others: number of \
            layer1 nodes is less than 1.\n")
    if len(graph.nodes) < 2:
        raise ValueError("Error connecting the node from others: there are no \
            other nodes in the graph.\n")

    s_node = random.randint(starting_layer1_node,
                     starting_layer1_node + number_of_layer1_nodes - 1)
    graph.add_edge(s_node, node)

"""Vertex distribution and max_edge_weights must be arrays."""
def generate_graph(n, vertex_distribution, max_edge_weight=100):
    DG = nx.DiGraph()

    """Distribute the vertices."""
    layers = _distribute_vertices(n, vertex_distribution)

    """Assign the number of the first vertex in a layer."""
    first_vertex = [0] * len(layers)
    for i in range(1, len(layers)):
        first_vertex[i] = first_vertex[i - 1] + layers[i - 1]

    """Add vertices to the graph."""
    vertices = []
    for i in range(n):
        vertices.append(i)
    DG.add_nodes_from(vertices)

    """Ensure that the graph is connected.
       Layer 0 nodes must have at least one outgoing edge.
       Layer 1-3 nodes must have at least one incoming edge and at least one
       outgoing edge.
       Layer 4 nodes must have at least one incoming edge."""
    for node in range(first_vertex[len(layers) - 1]):
        if not _is_outgoing_connectivity(DG, node):
            for layer in range(0, len(layers) - 1):
                if node in range(first_vertex[layer], first_vertex[layer + 1]):
                    _connect_to_anything(DG, node, first_vertex[layer + 1],
                                        layers[layer + 1])


    for node in range(first_vertex[1], n):
        if not _is_incoming_connectivity(DG, node):
            for layer in range(len(layers) - 1, 0, -1):
                if node in range(first_vertex[layer],
                                 first_vertex[layer] + layers[layer]):
                    _connect_from_anything(DG, node, first_vertex[layer - 1],
                                        layers[layer - 1])

    """Assign weights to the edges. For the first layer edges, generate the
    weights randomly. Then, additive model."""
    for node in range(layers[0]):
        for edge in DG.out_edges(node):
            DG.get_edge_data(*edge)['capacity'] = random.randint(1, max_edge_weight)
    for layer_number in range(1, len(layers) - 1):
        for node in range(first_vertex[layer_number], first_vertex[layer_number + 1]):
            out_weight = 0
            for u, v, data in DG.in_edges(node, data=True):
                out_weight += data['capacity']
            for edge in DG.out_edges(node):
                DG.get_edge_data(*edge)['capacity'] = out_weight

    """Assign attributes to the graph:
       number of nodes,
       number of nodes on the first layer,
       number of nodes on the last layer."""
    DG.graph["number_of_nodes"] = n
    DG.graph["number_of_first_layer_nodes"] = layers[0]
    DG.graph["number_of_last_layer_nodes"] = layers[len(layers) - 1]

    return DG

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

"""Calculate the exact utility of a graph."""
def get_utility(graph, graph_size):
    graph_utility = 0
    start_node = int(graph_size - 50)
    for sink in range(start_node, graph_size):
        if sink in graph.nodes:
            for edge in graph.in_edges(sink, data=True):
                graph_utility += graph.get_edge_data(*edge)['capacity']
    return graph_utility

def save_graph_to_file(graph, path_length, iteration):
    filename = "graphs/path_graph_" + str(path_length) + "_" + str(iteration) + ".json"
    nx.write_edgelist(graph, filename)

def extend_graph(graph, old_graph_size, new_graph_size):
    vertices = []
    for i in range(old_graph_size, new_graph_size):
        vertices.append(i)
    graph.add_nodes_from(vertices)

    for node in range(old_graph_size - 50, old_graph_size):
        c = graph.in_degree(node, "capacity")
        graph.add_edge(node, node + 50, capacity=c)
    return graph

def extend_constraints(old_constraints):
    new_constraints = []
    for s, t in old_constraints:
        new_constraints.append([s, t + 50])
    return new_constraints

"""Arguments: max_path_length, max_iterations."""
def main(argv):
    max_path_length = argv[0]
    max_iterations = argv[1]
    number_of_constraints = 10

    for iteration in range(1, int(max_iterations) + 1, 1):
        vertex_distribution = [50, 50, 50]
        path_length = 2
        graph_size = 150

        graph = generate_graph(graph_size, vertex_distribution)
        constraints = get_random_constraints(number_of_constraints, graph_size, graph, vertex_distribution)

        save_graph_to_file(graph, path_length, iteration)
        with open("constraints/path_cons_" + str(path_length) + "_" + str(iteration) + ".json", "w") as myfile:
            json.dump(list(constraints), myfile)

        for _ in range(2, int(max_path_length) - 1):
            path_length += 1
            vertex_distribution.append(50)
            graph = extend_graph(graph, graph_size, graph_size + 50)
            graph_size += 50
            constraints = extend_constraints(constraints)
            save_graph_to_file(graph, path_length, iteration)

            with open("constraints/path_cons_" + str(path_length) + "_" + str(iteration) + ".json", "w") as myfile:
                json.dump(list(constraints), myfile)

if __name__ == "__main__":
   main(sys.argv[1:])
