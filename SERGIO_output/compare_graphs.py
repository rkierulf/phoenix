import networkx as nx
import pandas as pd

file_path_inferred = 'inferred_graph.txt'
file_path_validation = '../SERGIO_data/validation_network.csv'
df_inferred = pd.read_csv(file_path_inferred, skiprows=1, header=None)
df_validation = pd.read_csv(file_path_validation, skiprows=1, header=None)

G_inferred = G = nx.from_pandas_edgelist(df_inferred, 0, 1, create_using=nx.Graph())
G_validation = nx.from_pandas_edgelist(df_validation, 0, 1, create_using=nx.Graph())

#Note: this will take a while
distance = nx.graph_edit_distance(G_inferred, G_validation, timeout=3000)
print("Calculated graph edit distance: " + str(distance))