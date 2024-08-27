import numpy as np

def generate_and_assign_biases_thresholds(layers):
    biases = {}
    thresholds = {}
    
    for i in range(len(layers) - 1):
        bias_variable_name = f'b_{i}_{i+1}'
        threshold_variable_name = f'T_{i}_{i+1}'

        current_layer_nodes = layers[i]
        next_layer_nodes = layers[i+1]
        num_connections = current_layer_nodes * next_layer_nodes
        
        print(f"Enter {num_connections} bias values for connections from layer {i+1} to layer {i+2} (comma separated):")
        bias_value = np.array([float(x) for x in input().split(',')]).reshape(current_layer_nodes, next_layer_nodes)
        
        print(f"Enter {num_connections} threshold values for connections from layer {i+1} to layer {i+2} (comma separated):")
        threshold_value = np.array([float(x) for x in input().split(',')]).reshape(current_layer_nodes, next_layer_nodes)
        
        biases[bias_variable_name] = bias_value
        thresholds[threshold_variable_name] = threshold_value
        
    return biases, thresholds

def custom_sigmoid(x, alpha=5):
    x = np.clip(x, -500/alpha, 500/alpha)
    return 1 / (1 + np.exp(-alpha * x))

def custom_sigmoid_derivative(x, alpha=20):
    sig = custom_sigmoid(x, alpha)
    return alpha * sig * (1 - sig)

def relu(x):
    return np.maximum(0, x)

def calculate_error(Y, T):
    return 0.5 * np.sum((Y - T) ** 2)

def forward_propagation(input_data, biases, thresholds, layers):
    activations = []  # To store activations of each layer
    Z = np.zeros((input_data.shape[0], layers[-1]))  # Final output layer

    # Calculate Zo based on the number of input layer nodes
    input_layer_nodes = layers[0]
    if input_layer_nodes == 2:
        Zo = 0.5
    elif input_layer_nodes == 3:
        Zo = 0.33
    elif input_layer_nodes == 4:
        Zo = 0.25
    else:
        Zo = 0.2

    print("\n=== Forward Propagation ===")
    print(f"Input data shape: {input_data.shape}")

    for i in range(input_data.shape[0]):
        A = input_data[i].reshape(-1, 1)
        activations.append(A)  # Store input layer activation
        print(f"Layer 0 (Input) activation shape: {A.shape}")

        for layer in range(len(layers) - 1):
            bias_variable_name = f'b_{layer}_{layer+1}'
            threshold_variable_name = f'T_{layer}_{layer+1}'

            bias = biases[bias_variable_name]
            threshold = thresholds[threshold_variable_name]

            weighted_sum_i = np.sum(custom_sigmoid(2 * bias - 1) * (A - threshold), axis=0)
            sum_i = np.sum(weighted_sum_i, axis=0)

            intermediate_value = Zo * (sum_i - input_data.shape[1] + layers[-1])
            A = relu(intermediate_value)
            A = np.atleast_2d(A)  # Ensure A is always 2D
            activations.append(A)  # Store activation for this layer
            print(f"Layer {layer + 1} activation shape: {A.shape}")

        Z[i] = A  # Final output for sample i

    print(f"Final output Z shape: {Z.shape}")
    print("=== End of Forward Propagation ===\n")

    return Z, Zo, activations

def calculate_beta(input_data, bias, threshold, Zo):
    print(f"input_data shape: {np.shape(input_data)}")  # Debug print
    print(f"bias shape: {bias.shape}")  # Debug print
    print(f"threshold shape: {threshold.shape}")  # Debug print

    # Ensure input_data is at least 1D
    if np.isscalar(input_data):
        input_data = np.array([input_data])
    
    # Ensure input_data is 2D
    if input_data.ndim == 1:
        input_data = input_data.reshape(-1, 1)

    beta_values = np.zeros(bias.shape)
    for i in range(bias.shape[0]):
        for j in range(bias.shape[1]):
            z = 2 * bias[i, j] - 1
            print(f"z[{i},{j}]:", z, "Shape:", np.shape(z))  # Debug print

            # Use input_data[0, 0] for scalar inputs
            input_value = input_data[0, 0] if input_data.size == 1 else input_data[i, 0]
            sigmoid_input = z * (input_value - threshold[i, j])
            print(f"sigmoid_input[{i},{j}]:", sigmoid_input, "Shape:", np.shape(sigmoid_input))  # Debug print

            g_prime = custom_sigmoid_derivative(sigmoid_input)
            print(f"g_prime[{i},{j}]:", g_prime, "Shape:", np.shape(g_prime))  # Debug print

            beta_values[i, j] = Zo * g_prime * z
            print(f"beta_values[{i},{j}] = {beta_values[i, j]}\n")  # Debug print

    return beta_values

def calculate_theta(input_data, bias, threshold, Zo):
    if np.isscalar(input_data):
        input_data = np.array([input_data])
    if input_data.ndim == 1:
        input_data = input_data.reshape(-1, 1)  # Ensure it's 2D

    theta_values = np.zeros(bias.shape)
    for i in range(bias.shape[0]):
        for j in range(bias.shape[1]):
            z = 2 * bias[i, j] - 1
            input_value = input_data[0, 0] if input_data.size == 1 else input_data[i, 0]
            sigmoid_input = z * (input_value - threshold[i, j])
            g_prime = custom_sigmoid_derivative(sigmoid_input)
            theta_values[i, j] = Zo * g_prime * 2 * (input_value - threshold[i, j])

    return theta_values


def back_propagation(activations, biases, thresholds, layers, Zo):
    print("\n=== Backpropagation ===")

    for layer in range(len(layers) - 1):
        A_prev = activations[layer]
        A_curr = activations[layer + 1]

        # Ensure A_prev is 2D
        if A_prev.ndim == 1:
            A_prev = A_prev.reshape(-1, 1)

        bias_variable_name = f'b_{layer}_{layer+1}'
        threshold_variable_name = f'T_{layer}_{layer+1}'

        bias = biases[bias_variable_name]
        threshold = thresholds[threshold_variable_name]

        # Calculate beta and theta using activations
        beta = calculate_beta(A_prev, bias, threshold, Zo)
        theta = calculate_theta(A_prev, bias, threshold, Zo)

        print(f"\nLayer {layer + 1} -> Layer {layer + 2}:")
        print(f"Beta values:\n{beta}")
        print(f"Theta values:\n{theta}")


def main():
    num_hidden_layers = int(input("Enter the number of hidden layers: "))
    hidden_layers = []
    for i in range(num_hidden_layers):
        nodes = int(input(f"Enter the number of nodes in hidden layer {i+1}: "))
        hidden_layers.append(nodes)
    
    print("Number of nodes in each hidden layer is:", hidden_layers)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])
    
    n_input_nodes = X.shape[1]
    n_output_nodes = y.shape[1]
    
    layers = [n_input_nodes] + hidden_layers + [n_output_nodes]
    
    biases, thresholds = generate_and_assign_biases_thresholds(layers)
    
    print("Generated biases:", biases)
    print("Generated thresholds:", thresholds)

    epochs = 1
    

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1} ===")
        Y_pred, Zo, activations = forward_propagation(X, biases, thresholds, layers)
        error = calculate_error(Y_pred, y)
        back_propagation(activations, biases, thresholds, layers, Zo)
        print(f"Epoch {epoch + 1}, Error: {error}")

    print("Final biases:", biases)
    print("Final thresholds:", thresholds)

    #Y_pred, Zo = forward_propagation(X, biases, thresholds, layers)
    print("Final Output:", Y_pred)

if __name__ == "__main__":
    main()
