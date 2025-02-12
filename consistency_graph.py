import sqlite3
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
def pair_id_to_image_ids(pair_id, num_images):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2


def random_walk_node(G, node, node_count):
    neighbors = list(G.neighbors(node))  # Get adjacent nodes
    if not neighbors:
        return None  # No adjacent nodes

    # Get the edge weights
    weights = [(G[node][neighbor].get('weight', 1)/node_count[neighbor]) for neighbor in neighbors]

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # Choose a neighbor based on probabilities
    chosen_node = random.choices(neighbors, weights=probabilities)[0]
    return chosen_node

def load_consistency_graph(colmap_database_path):
    # Connect to COLMAP database
    db_path = os.path.join(colmap_database_path + "database.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the number of images
    cursor.execute("SELECT COUNT(*) FROM images;")
    num_images = cursor.fetchone()[0]

    # Query the co-visibility graph
    cursor.execute("SELECT pair_id, rows FROM two_view_geometries;")
    pairs = cursor.fetchall()

    # Build the graph
    G = nx.Graph()
    for pair_id, matches in pairs:
        img1, img2 = pair_id_to_image_ids(pair_id, num_images)
        if matches > 0:
            G.add_edge(img1, img2, weight=matches)
    return G