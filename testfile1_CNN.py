import numpy as np

import numpy as np

def generate_and_assign_biases_thresholds(layers):
    biases = {}
    thresholds = {}
    
    for i in range(len(layers) - 1):
        bias_variable_name = f'b_{i}_{i+1}'
        threshold_variable_name = f'T_{i}_{i+1}'

        current_layer_nodes = layers[i]
        next_layer_nodes = layers[i+1]

        print(f"Enter {current_layer_nodes * next_layer_nodes} bias values for connections from layer {i+1} to layer {i+2} (comma separated):")
        bias_value = np.array([float(x) for x in input().split(',')]).reshape(next_layer_nodes, current_layer_nodes)
        
        print(f"Enter {current_layer_nodes * next_layer_nodes} threshold values for connections from layer {i+1} to layer {i+2} (comma separated):")
        threshold_value = np.array([float(x) for x in input().split(',')]).reshape(next_layer_nodes, current_layer_nodes)
        
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

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def calculate_error(Y, T):
    return 0.5 * np.sum((Y - T) ** 2)

def forward_propagation(input_data, biases, thresholds, layers):
    N = input_data.shape[1]  # Number of features

    output_layer_nodes = layers[-1]
    input_layer_nodes = layers[0]

    # Define Zo based on the number of output nodes
    if input_layer_nodes == 2:
        Zo = 0.5
    elif input_layer_nodes == 3:
        Zo = 0.33
    elif input_layer_nodes == 4:
        Zo = 0.25
    else:  # Assuming that for 5 or more nodes, we use the same scaling factor
        Zo = 0.2

    Z = np.zeros((input_data.shape[0], output_layer_nodes))

    # Compute Z for each sample
    for i in range(input_data.shape[0]):  # Loop through samples
        print("loop no:", i)
        A = input_data[i].reshape(-1,1)  # Initialize activation for input layer
        
        for layer in range(len(layers) - 1):
            bias_variable_name = f'b_{layer}_{layer+1}'
            threshold_variable_name = f'T_{layer}_{layer+1}'
            
            bias = biases[bias_variable_name]
            threshold = thresholds[threshold_variable_name]

            # Ensure A is reshaped to match the size of the current layer
            # print("A before shape config:", A)
            # if A.shape[0] != threshold.shape[0]:
            #     A = np.tile(A, (threshold.shape[0], 1)).T[0]
            #     print("A after shape config:",A)
            
            # Apply custom sigmoid to each bias and subtract the threshold for each feature
            weighted_sum_i = np.sum(custom_sigmoid(2 * bias - 1) * (A - threshold))
            print(f"weighted_sum for layer {layer}: {weighted_sum_i}")
            sum_i = np.sum(weighted_sum_i, axis=0)
            print("sum_i", sum_i)
            print("N", N)
            print("output_layer_nodes", output_layer_nodes)
            
            # Calculate Z based on the provided formula
            intermediate_value = Zo * (sum_i - N + output_layer_nodes)
            print("intermediate_value", intermediate_value)
            
            # applying Relu for next layer
            A = relu(intermediate_value)
            print(f"A after relu: {A}")

        Z[i] = A.flatten()  # Final output for sample i
        print(f"Final output Z[{i}]: {Z[i]}")  
            
    return Z, Zo

def calculate_beta(input_data, bias, threshold, Zo):
    beta_values = np.zeros(input_data.shape)
    for i in range(input_data.shape[0]):
        for k in range(input_data.shape[1]):
            if k >= bias.shape[0]:  # Ensure k does not exceed bias length
                continue

            z = 2 * bias[k] - 1
            sigmoid_input = z * (input_data[i, k] - threshold[k])
            g_prime = custom_sigmoid_derivative(sigmoid_input)

            # Handle g_prime and ensure it's scalar
            if g_prime.size > 1:
                g_prime = g_prime.mean()
            else:
                g_prime = g_prime.item()

            beta_value = Zo * g_prime * z
            beta_values[i, k] = beta_value

    return beta_values






def calculate_delta_threshold(Y, T, Z, Beta):
    return np.where(Z > 0, (T - Y) * Beta, 0)

def calculate_delta_bias(Y, T, Z, Theta):
    return np.where(Z > 0, (Y - T) * Theta, 0)


def calculate_theta(input_data, bias, threshold, Zo):
    if len(input_data.shape) == 1:
        input_data = np.expand_dims(input_data, axis=-1)

    if input_data.shape[1] != bias.shape[0]:
        input_data = np.tile(input_data, (1, bias.shape[0]))

    theta_values = np.zeros((input_data.shape[0], bias.shape[0]))
    for i in range(input_data.shape[0]):
        for k in range(bias.shape[0]):
            z = 2 * bias[k] - 1
            scalar_input = input_data[i, k] - threshold[k]
            sigmoid_input = z * scalar_input
            
            g_prime = custom_sigmoid_derivative(sigmoid_input)

            if g_prime.size > 1:
                g_prime = g_prime.mean()

            scalar_value = 2 * scalar_input
            theta_values[i, k] = Zo * g_prime * scalar_value

    return theta_values



# def calculate_delta_bias(Y, T, Z, Theta):
#     Delta_B = np.where(Z > 0, (Y - T) * Theta, 0)
#     return Delta_B


def main():
    num_hidden_layers = int(input("Enter the number of hidden layers: "))
    hidden_layers = []
    for i in range(num_hidden_layers):
        nodes = int(input(f"Enter the number of nodes in hidden layer {i+1}: "))
        hidden_layers.append(nodes)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])
    
    n_input_nodes = X.shape[1]
    n_output_nodes = y.shape[1]
    
    layers = [n_input_nodes] + hidden_layers + [n_output_nodes]
    
    biases, thresholds = generate_and_assign_biases_thresholds(layers)

    epochs = 5
    
    for epoch in range(epochs):
        Y_pred, Zo = forward_propagation(X, biases, thresholds, layers)
        error = calculate_error(Y_pred, y)
        print(f"Epoch {epoch + 1}, Error: {error}")

        for layer in range(len(layers) - 2, -1, -1):
            bias_variable_name = f'b_{layer}_{layer+1}'
            threshold_variable_name = f'T_{layer}_{layer+1}'
    
            if layer == len(layers) - 2:
                Z = Y_pred
                Theta = calculate_theta(Z, biases[bias_variable_name], thresholds[threshold_variable_name], Zo)
                Beta = calculate_beta(Z, biases[bias_variable_name], thresholds[threshold_variable_name], Zo)
            else:
                Z, Zo = forward_propagation(X, biases, thresholds, layers[:layer+2])
                Theta = calculate_theta(Z, biases[bias_variable_name], thresholds[threshold_variable_name], Zo)
                Beta = calculate_beta(Z, biases[bias_variable_name], thresholds[threshold_variable_name], Zo)

            delta_threshold = calculate_delta_threshold(Y_pred, y, Z, Beta)
            delta_bias = calculate_delta_bias(Y_pred, y, Z, Theta)
            
            if delta_threshold.size == thresholds[threshold_variable_name].size:
                thresholds[threshold_variable_name] -= delta_threshold.mean(axis=0).reshape(thresholds[threshold_variable_name].shape)
            else:
                thresholds[threshold_variable_name] -= delta_threshold.mean(axis=0)

            if delta_bias.size == biases[bias_variable_name].size:
                biases[bias_variable_name] -= delta_bias.mean(axis=0).reshape(biases[bias_variable_name].shape)
            else:
                biases[bias_variable_name] -= delta_bias.mean(axis=0)

    print("Output:", Y_pred)

if __name__ == "__main__":
    main()


 