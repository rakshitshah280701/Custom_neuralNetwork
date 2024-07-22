import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def custom_sigmoid(x, alpha=5):
    x = np.clip(x, -500/alpha, 500/alpha)
    return 1 / (1 + np.exp(-alpha * x))

def relu(x):
    return np.maximum(0, x)

def visualize_forward_propagation(input_data, biases, thresholds, layers):
    activations = [input_data]  # List to store activations of each layer
    Zo_values = []  # List to store Zo values for each layer

    G = nx.DiGraph()
    pos = {}
    
    node_idx = 0
    layer_nodes = []
    
    # Add input layer nodes
    input_layer = []
    for i in range(layers[0]):
        G.add_node(node_idx, layer=0)
        pos[node_idx] = (0, -i)
        input_layer.append(node_idx)
        node_idx += 1
    layer_nodes.append(input_layer)

    # Add hidden and output layer nodes
    for layer in range(1, len(layers)):
        layer_nodes.append([])
        for i in range(layers[layer]):
            G.add_node(node_idx, layer=layer)
            pos[node_idx] = (layer, -i)
            layer_nodes[-1].append(node_idx)
            node_idx += 1

    # Add edges between layers
    for layer in range(len(layers) - 1):
        for src in layer_nodes[layer]:
            for dst in layer_nodes[layer + 1]:
                G.add_edge(src, dst)
    
    for layer in range(len(layers) - 1):
        bias_variable_name = f'b_{layer}_{layer+1}'
        threshold_variable_name = f'T_{layer}_{layer+1}'
        
        bias = biases[bias_variable_name]
        threshold = thresholds[threshold_variable_name]

        previous_activations = activations[-1]
        current_layer_nodes = layers[layer+1]
        
        Z = np.zeros((previous_activations.shape[0], current_layer_nodes))

        for j in range(current_layer_nodes):  # Loop through nodes in the current layer
            weighted_sum = np.dot(previous_activations, bias[:, j]) - threshold[j]
            Z[:, j] = custom_sigmoid(2 * weighted_sum - 1)
        
        A = relu(Z)
        activations.append(A)
        
        # Calculate Zo for each layer
        input_layer_nodes = layers[0]
        if input_layer_nodes == 2:
            Zo = 0.5
        elif input_layer_nodes == 3:
            Zo = 0.33
        elif input_layer_nodes == 4:
            Zo = 0.25
        else:
            Zo = 0.2
        
        Zo_values.append(Zo)
    
    # Draw the network
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', ax=ax)
    plt.title("Dynamic Forward Propagation Visualization")
    plt.show()
    
    return activations, Zo_values

# Example to check if the function works correctly
layers = [2, 3, 1]  # Input layer with 2 nodes, one hidden layer with 3 nodes, output layer with 1 node
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

biases = {
    'b_0_1': np.random.rand(2, 3),
    'b_1_2': np.random.rand(3, 1)
}

thresholds = {
    'T_0_1': np.random.rand(3),
    'T_1_2': np.random.rand(1)
}

activations, Zo_values = visualize_forward_propagation(input_data, biases, thresholds, layers)
print("Activations:", activations)
print("Zo values:", Zo_values)
