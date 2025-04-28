
import sqlite3
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Not strictly needed in newer versions, but safe to include
from sklearn.neighbors import NearestNeighbors
import consistency_graph
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def construct_distance_graph(images_file):
    with open(images_file) as file:
        lines = [line.rstrip() for line in file]
        positions = np.zeros((len(lines), 3))
        quats = np.zeros((len(lines), 4))
        for i, line in enumerate(lines):
            split = line.split(" ")
            positions[i, 0] = split[5]
            positions[i, 1] = split[6]
            positions[i, 2] = split[7]
            quats[i, 0] = split[1]
            quats[i, 1] = split[2]
            quats[i, 2] = split[3]
            quats[i, 3] = split[4]
        print(positions.shape)
        
        for i in range(len(lines)):
            positions[i] = qvec2rotmat(quats[i]).transpose() @ positions[i]
        positions[i, 2] /= 8
        
        points = positions

        k = 1000
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)

        G_knn = nx.DiGraph()

        # Add nodes
        for i in range(points.shape[0]):
            G_knn.add_node(i, pos=tuple(points[i]))

        # Add edges (skip the first neighbor since it's the point itself)
        for i in range(points.shape[0]):
            for j in range(1, k+1):
                neighbor_idx = indices[i][j]
                dist = distances[i][j]
                G_knn.add_edge(i, neighbor_idx, weight=float(1000.0/(np.sqrt(dist) + 15)))
                
        nx.write_edgelist(G_knn, "/home/felix-windisch/Datasets/BIGCITY/camera_calibration/aligned/sparse/0/consistency_graph.edge_list")
        print(G_knn.nodes)
        node = 0
        N = 5000
        colors = np.zeros((N, 3))
        walk = np.zeros(N, dtype=np.int32)
        current_color = np.random.rand(3)
        for i in range(N):
            colors[i] = current_color
            node = consistency_graph.metropolis_hastings_walk(G_knn, node)
            walk[i] = node
            if i % 100 == 0:
                i = random.randint(0, points.shape[0])
                current_color = np.random.rand(3)
        neighbours = walk #np.array(list(nx.all_neighbors(G_knn, 5000)))
        print(neighbours)
        print(G_knn.number_of_edges())
        x = positions[neighbours, 0]
        y = positions[neighbours, 1]
        z = positions[neighbours, 2]*10
        # Create 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
       #ax.plot(x, y, z, color='blue', label='Path')
        ax.scatter(x, y, z, c=colors, marker='o')
        # Optional: Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('equal')
        plt.show()



construct_distance_graph("/home/felix-windisch/Datasets/BIGCITY/camera_calibration/aligned/sparse/0/images.txt")


