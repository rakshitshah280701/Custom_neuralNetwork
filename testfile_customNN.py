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
    N = input_data.shape[1]  # Number of features

    output_layer_nodes = layers[-1]
    input_layer_nodes = layers[0]

    if input_layer_nodes == 2:
        Zo = 0.5
    elif input_layer_nodes == 3:
        Zo = 0.33
    elif input_layer_nodes == 4:
        Zo = 0.25
    else:
        Zo = 0.2

    Z = np.zeros((input_data.shape[0], output_layer_nodes))

    for i in range(input_data.shape[0]):
        A = input_data[i].reshape(-1, 1)

        print(f"\n--- Forward Propagation for Input Sample {i + 1} ---")
        print(f"Initial input A: {A.flatten()}")
        
        for layer in range(len(layers) - 1):
            bias_variable_name = f'b_{layer}_{layer+1}'
            threshold_variable_name = f'T_{layer}_{layer+1}'
            
            bias = biases[bias_variable_name]
            threshold = thresholds[threshold_variable_name]

            print(f"\nLayer {layer + 1} -> Layer {layer + 2}:")
            print(f"Biases:\n{bias}")
            print(f"Thresholds:\n{threshold}")

            weighted_sum_i = np.sum(custom_sigmoid(2 * bias - 1) * (A - threshold), axis=0)
            sum_i = np.sum(weighted_sum_i, axis=0)
            print(f"Weighted sum for layer {layer + 1}: {weighted_sum_i}")
            print(f"Sum across neurons: {sum_i}")


            intermediate_value = Zo * (sum_i - N + output_layer_nodes)
            print(f"Intermediate value before ReLU: {intermediate_value}")

            A = relu(intermediate_value)
            print("Value after relu:",A)

        Z[i] = A  # Final output for sample i
        print(f"Final output Z[{i}]: {Z[i]}")  

    return Z, Zo


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
        Y_pred, Zo = forward_propagation(X, biases, thresholds, layers)
        error = calculate_error(Y_pred, y)
        print(f"Epoch {epoch + 1}, Error: {error}")

    print("Final biases:", biases)
    print("Final thresholds:", thresholds)

    #Y_pred, Zo = forward_propagation(X, biases, thresholds, layers)
    print("Final Output:", Y_pred)

if __name__ == "__main__":
    main()
