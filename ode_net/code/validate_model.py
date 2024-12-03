import numpy as np
import os

def compare_graphs(true_graph, other_graph, percentiles_mat, genes_list):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(percentiles_mat.shape[0]):
        for j in range(percentiles_mat.shape[0]):
            if (i != j):
                gene1 = genes_list[i]
                gene2 = genes_list[j]
                edge_in_true_graph = gene1 in true_graph and true_graph[gene1].count(gene2) > 0
                edge_in_other_graph = gene1 in other_graph and other_graph[gene1].count(gene2) > 0
                if edge_in_true_graph and edge_in_other_graph:
                    tp += 1
                elif (not edge_in_true_graph) and (not edge_in_other_graph):
                    tn += 1
                elif (not edge_in_true_graph) and (edge_in_other_graph):
                    fp += 1
                elif (edge_in_true_graph) and (not edge_in_other_graph):
                    fn += 1
                
    return tp, tn, fp, fn

def get_edges(percentiles_mat, c, genes_list):
    edges = {}
    for i in range(percentiles_mat.shape[0]):
        for j in range(percentiles_mat.shape[0]):
            if percentiles_mat[i,j] >= c and (i != j):
                gene1 = genes_list[i]
                gene2 = genes_list[j]
                if gene1 not in edges:
                    edges[gene1] = []
                edges[gene1].append(gene2)
    return edges

def validate_model(output_root_dir, genes_path, validation_network_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gene_names_path = os.path.join(current_dir, genes_path)
    genes_list = []
    genes_dict = {}
    with open(gene_names_path, 'r') as genes_file:
        #Skip the first line (header)
        next(genes_file)
        for line in genes_file:
            gene_name = line.strip().replace('"','')
            genes_list.append(gene_name)
            genes_dict[gene_name] = True
    
    genes_file.close()

    edges_path = os.path.join(current_dir, validation_network_path)
    edges_dict = {}
    with open(edges_path, 'r') as validation_graph_file:
        #Skip the first line (header)
        next(validation_graph_file)
        for line in validation_graph_file:
            gene1, gene2 = line.strip().replace('"','').split(",")
            if (gene1 in genes_dict) and (gene2 in genes_dict):
                if gene1 in edges_dict:
                    edges_dict[gene1].append(gene2)
                else:
                    edges_dict[gene1] = [gene2]

    validation_graph_file.close()
    percentiles_mat_path = os.path.join(current_dir, output_root_dir, 'dynamics_mat.csv')
    percentiles_mat = np.loadtxt(percentiles_mat_path, delimiter=',')

    #From section 2.2 of https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-024-03264-0/MediaObjects/13059_2024_3264_MOESM2_ESM.pdf
    c_values = np.linspace(0,1,101)
    c_val = -1
    best_accuracy = -1
    best_graph = None
    for i in range(1, 101):
        c = c_values[i]
        inferred_edges = get_edges(percentiles_mat, c, genes_list)
        tp, tn, fp, fn = compare_graphs(edges_dict, inferred_edges, percentiles_mat, genes_list)
        if (tp + fn) > 0:
            tpr = tp / (tp + fn)
        else:
            tpr = 0
        if (tn + fp) > 0:
            tnr = tn / (tn + fp)
        else:
            tnr = 0
        accuracy = (tpr + tnr) / 2
        if (accuracy > best_accuracy):
            best_graph = inferred_edges
            best_accuracy = accuracy
            c_val = c
    
    print("Chose value of C: " + str(c_val))
    print("Balanced classification accuracy: " + str(best_accuracy))
    
    inferred_graph_path = os.path.join(current_dir, output_root_dir, "inferred_graph.txt")
    with open(inferred_graph_path, 'w') as inferred_graph_file:
        inferred_graph_file.write("\"from\",\"to\"\n")
        for edge1 in best_graph:
            for edge2 in best_graph[edge1]:
                inferred_graph_file.write(f"\"{edge1}\",\"{edge2}\"\n") 


    