import numpy as np

def generate_and_assign_biases_thresholds(layers):
    biases = {}
    thresholds = {}
    
    for i in range(len(layers) - 1):
        bias_variable_name = f'b_{i}_{i+1}'
        threshold_variable_name = f'T_{i}_{i+1}'
        
        bias_value = np.random.uniform(0.1, 0.9, size=layers[i+1])
        threshold_value = np.random.uniform(0.1, 0.9, size=layers[i+1])
        
        biases[bias_variable_name] = bias_value
        thresholds[threshold_variable_name] = threshold_value
        
    return biases, thresholds

def custom_sigmoid(x, alpha=5):
    x = np.clip(x, -500/alpha, 500/alpha)
    return 1 / (1 + np.exp(-alpha * x))

def custom_sigmoid_derivative(x, alpha=20):
    sig = custom_sigmoid(x, alpha)
    return alpha * sig * (1 - sig)

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
            
            # Apply custom sigmoid to each bias and subtract the threshold for each feature
            sum_i = np.sum(custom_sigmoid(2 * bias - 1) * (A - threshold))
            print("sum_i", sum_i)
            print("N", N)
            print("output_layer_nodes", output_layer_nodes)
            
            # Calculate Z based on the provided formula
            Z[i] = Zo * (sum_i - N + output_layer_nodes)
            print("value of Zo for this iteration is:", Zo)
            print(f"value of i is: {i}, the value for Z: {Z[i]}")
            
            # Update activation for next layer
            A = custom_sigmoid(sum_i)
            
    return Z, Zo

def relu(x):
    return np.maximum(0, x)

def calculate_error(Y, T):
    return 0.5 * np.sum((Y - T) ** 2)

def calculate_beta(input_data, bias, threshold,Zo):

    beta_values = np.zeros(input_data.shape)
    for i in range(input_data.shape[0]):
        for k in range(input_data.shape[1]):
            z = 2 * bias[k] - 1
            # Compute the input for the sigmoid derivative
            sigmoid_input = z * (input_data[i, k] - threshold[k])
            # Get the derivative of the custom sigmoid at the computed input
            g_prime = custom_sigmoid_derivative(sigmoid_input)
            # Calculate the beta value
            beta_values[i, k] = Zo * g_prime * z

    return beta_values

def calculate_delta_threshold(Y, T, Z, Beta):
    Delta_T = np.where(Z > 0, (T - Y) * Beta, 0)
    return Delta_T

def calculate_theta(input_data, bias, threshold,Zo):
  # Ensure input_data has the correct shape
    if len(input_data.shape) == 1:
        # If input_data is 1D (only one neuron's output), expand dims to match bias/threshold lengths
        input_data = np.expand_dims(input_data, axis=-1)

    # Adjust to broadcast input_data if it's for a single neuron to multiple outputs
    if input_data.shape[1] != len(bias):
        # Assuming input_data comes from a layer with fewer neurons, replicate it to match the number of biases
        input_data = np.tile(input_data, (1, len(bias)))


    theta_values = np.zeros((input_data.shape[0], len(bias)))
    for i in range(input_data.shape[0]):
        for k in range(len(bias)):  # For each neuron
            z = 2 * bias[k] - 1
            sigmoid_input = z * (input_data[i, k] - threshold[k])
            g_prime = custom_sigmoid_derivative(sigmoid_input)
            theta_values[i, k] = Zo * g_prime  * 2 * (input_data[i, k] - threshold[k])

    return theta_values

def calculate_delta_bias(Y, T, Z, Theta):
    Delta_B = np.where(Z > 0, (Y - T) * Theta, 0)
    return Delta_B



X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

def main():
    # User input for the number of hidden layers and nodes in each hidden layer
    num_hidden_layers = int(input("Enter the number of hidden layers: "))
    hidden_layers = []
    for i in range(num_hidden_layers):
        nodes = int(input(f"Enter the number of nodes in hidden layer {i+1}: "))
        hidden_layers.append(nodes)
    
    print("Number of nodes in each hidden layer is:", hidden_layers)
    
    # Example input data
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0],
                  [0],
                  [0],
                  [1]])
    
    n_input_nodes = X.shape[1]
    n_output_nodes = 2
    
    # Include input and output layers in the layer definition
    layers = [n_input_nodes] + hidden_layers + [n_output_nodes]
    
    # Generate biases and thresholds
    biases, thresholds = generate_and_assign_biases_thresholds(layers)
    
    # Print generated biases and thresholds for verification
    print("Generated biases:", biases)
    print("Generated thresholds:", thresholds)

    # Forward propagation
    output, Zo = forward_propagation(X, biases, thresholds, layers)
    print("Output:", output)


if __name__ == "__main__":
    main()