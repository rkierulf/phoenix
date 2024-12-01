import numpy as np
import os

def validate_model(output_root_dir, genes_path, validation_network_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gene_names_path = os.path.join(current_dir, genes_path)
    genes_list = []
    genes_dict = {}
    with open(gene_names_path, 'r') as genes_file:
        #Skip the first line (header)
        next(genes_file)
        for line in genes_file:
            gene_name = line.strip().strip('"')
            genes_list.append(gene_name)
            genes_dict[gene_name] = None
    
    genes_file.close()

    edges_path = os.path.join(current_dir, validation_network_path)
    print(edges_path)
    edges_dict = {}
    with open(edges_path, 'r') as validation_graph_file:
        #Skip the first line (header)
        next(validation_graph_file)
        for line in validation_graph_file:
            gene1, gene2 = line.split(',')
            gene1 = gene1.strip('"')
            gene2 = gene2.strip('"')
            if gene1 in genes_dict and gene2 in genes_dict:
                edges_dict[gene1] = gene2

    validation_graph_file.close()
    print(edges_dict)