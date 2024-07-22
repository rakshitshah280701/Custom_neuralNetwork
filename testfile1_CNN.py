import numpy as np

def generate_and_assign_biases_thresholds(layers):
    biases = {}
    thresholds = {}
    
    for i in range(len(layers) - 1):
        bias_variable_name = f'b_{i}_{i+1}'
        threshold_variable_name = f'T_{i}_{i+1}'

        # Number of connections from current layer to next layer
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
        A = input_data[i]  # Initialize activation for input layer
        
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
            sum_i = np.sum(custom_sigmoid(2 * bias - 1) * (A - threshold))
            print("sum_i", sum_i)
            print("N", N)
            print("output_layer_nodes", output_layer_nodes)
            
            # Calculate Z based on the provided formula
            intermediate_value = Zo * (sum_i - N + output_layer_nodes)
            print("value of Zo for this iteration is:", Zo)
            print(f"value of i is: {i}, the value for Z: {Z[i]}")
            
            # applying Relu for next layer
            Z[i] = relu(intermediate_value)
            A = Z[i]
            
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

    # epochs = 5

    # for epoch in range(epochs):
    #     Z_list = []
    #     Zo_list = []
    #     A = X

    #     # Forward propagation through all layers
    #     for i in range(len(layers) - 1):
    #         Z, Zo = forward_propagation(A, biases, thresholds, layers[i:i+2])
    #         Z_list.append(Z)
    #         Zo_list.append(Zo)
    #         A = Z
        
    #     Y0 = Z_list[-1]
    #     print(f"Epoch {epoch+1}, Y0: {Y0}")
    #     print(f"Epoch {epoch+1}, y: {y}")

    #     E = calculate_error(y, Y0)
        
    #     # Backpropagation
    #     for i in range(len(layers) - 2, -1, -1):
    #         if i == len(layers) - 2:
    #             Y = Z_list[i]
    #             Theta = calculate_theta(Z_list[i-1], biases[f'b_{i}_{i+1}'], thresholds[f'T_{i}_{i+1}'], Zo_list[i])
    #             Beta = calculate_beta(Z_list[i-1], biases[f'b_{i}_{i+1}'], thresholds[f'T_{i}_{i+1}'], Zo_list[i])
    #             delta_threshold = calculate_delta_threshold(Y0, y, Y, Beta)
    #             delta_bias = calculate_delta_bias(Y0, y, Y, Theta)
    #         else:
    #             Theta = calculate_theta(Z_list[i-1], biases[f'b_{i}_{i+1}'], thresholds[f'T_{i}_{i+1}'], Zo_list[i])
    #             Beta = calculate_beta(Z_list[i-1], biases[f'b_{i}_{i+1}'], thresholds[f'T_{i}_{i+1}'], Zo_list[i])
    #             delta_threshold = calculate_delta_threshold(Z_list[i+1], y, Z_list[i], Beta)
    #             delta_bias = calculate_delta_bias(Z_list[i+1], y, Z_list[i], Theta)

    #         thresholds[f'T_{i}_{i+1}'] -= delta_threshold.mean(axis=0)
    #         biases[f'b_{i}_{i+1}'] -= delta_bias.mean(axis=0)

    #     print(f'Epoch {epoch + 1}/{epochs}, Loss: {E}')

    Y_pred, Z2 = forward_propagation(X, biases, thresholds, layers)
    Y_pred1 = relu(Y_pred)
    
    # predicted_classes = Y_pred1 > 0.5
    # true_classes = y > 0.5
    # print("true_classes", true_classes)
    # accuracy = np.mean(predicted_classes == true_classes)
    # print(f'Model accuracy: {accuracy*100}%')

    # output, Zo = forward_propagation(X, biases, thresholds, layers)
    print("Output:", Y_pred1)

if __name__ == "__main__":
    main()