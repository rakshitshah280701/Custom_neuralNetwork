import numpy as np

def generate_and_assign_biases_thresholds(layer_node_counts):

    for i in range(len(layer_node_counts) - 1):
        # Create variable names
        bias_variable_name = f'b_{i}{i+1}'
        threshold_variable_name = f'T_{i}{i+1}'

        # Generate random biases and thresholds
        bias_value = np.random.uniform(0.1, 0.9, size=layer_node_counts[i])
        threshold_value = np.random.uniform(0.1, 0.9, size=layer_node_counts[i])

        # Assign to variables by updating the globals() dictionary (use with caution)
        globals()[bias_variable_name] = bias_value
        globals()[threshold_variable_name] = threshold_value

def custom_sigmoid(x, alpha=5):
    x = np.clip(x, -500/alpha, 500/alpha)
    return 1 / (1 + np.exp(-alpha * x))

def custom_sigmoid_derivative(x, alpha=20):
    sig = custom_sigmoid(x, alpha)
    return alpha * sig * (1 - sig)

def forward_propagation(input_data, biases, thresholds, output_layer_nodes,input_layer_nodes):
    N = input_data.shape[1]  # Number of features
    Z = np.zeros((input_data.shape[0], output_layer_nodes))

    # Define Zo based on the number of output nodes
    if input_layer_nodes == 2:
        Zo = 0.5
    elif input_layer_nodes == 3:
        Zo = 0.33
    elif input_layer_nodes == 4:
        Zo = 0.25
    else:  # Assuming that for 5 or more nodes, we use the same scaling factor
        Zo = 0.2

    # Compute Z for each sample
    for i in range(input_data.shape[0]):  # Loop through samples
        print("loop no:",i)
        # Apply custom sigmoid to each bias and subtract the threshold for each feature
        sum_i = np.sum(custom_sigmoid(2 * biases - 1) * (input_data[i] - thresholds))
        print("sum_i",sum_i)
        print("N",N)
        print("output_layer_nodes",output_layer_nodes)
        # Calculate Z based on the provided formula
        Z[i] = Zo * (sum_i - N + output_layer_nodes)
        print("value of Zo for this iteration is:",Zo)
        print(f"value of i is: {i}, the value for Z: {Z[i]}")

    return Z,Zo

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


n_input_nodes = X.shape[1]
n_hidden_nodes = 1
n_output_nodes = 2

b_01 = np.array([1.0, 1.0])
T_01 = np.array([0.5, 0.5])

b_12 = np.array([1.0, 1.0])
T_12 = np.array([0.6, 0.4])

epochs = 5
#learning_rate = 0.1

for epoch in range(epochs):
    Zh,Zo_h = forward_propagation(X, globals()['b_01'], globals()['T_01'], n_hidden_nodes,n_input_nodes)
    Yh = relu(Zh)
    print(f"Epoch {epoch+1}, Yh: {Yh}")
    Z0,Zo_0 = forward_propagation(X, globals()['b_01'], globals()['T_01'], n_output_nodes,n_hidden_nodes)
    Y0 = relu(Z0)
    print(f"Epoch {epoch+1}, Z0: {Z0}")
    print(f"Epoch {epoch+1}, Y0: {Y0}")
    print(f"Epoch {epoch+1}, y: {y}")



    E = calculate_error(y, Y0)

    # Backpropagation
    theta_ho = calculate_theta(Yh, globals()['b_12'], globals()['T_12'],Zo_0)
    print(f"Epoch {epoch+1}, thetha_ho: {theta_ho}")

    beta_ho = calculate_beta(Yh, globals()['b_12'], globals()['T_12'],Zo_0)
    print(f"Epoch {epoch+1}, beta_ho: {beta_ho}")


    delta_threshold_ho = calculate_delta_threshold(Y0, y, Z0, beta_ho)
    print(f"Epoch {epoch+1}, delta_threshold_ho: {delta_threshold_ho}")

    delta_bias_ho = calculate_delta_bias(Y0, y, Z0, theta_ho)
    print(f"Epoch {epoch+1}, delta_bias_ho: {delta_bias_ho}")


    # Update thresholds and biases for output layer
    globals()['T_12'] -=  delta_threshold_ho.mean(axis=0)
    print(f"Epoch {epoch+1}, T_12: {T_12}")

    globals()['b_12'] -=  delta_bias_ho.mean(axis=0)
    print(f"Epoch {epoch+1}, b_12: {b_12}")


    beta_ih = calculate_beta(X, globals()['b_01'], globals()['T_01'],Zo_h)
    print(f"Epoch {epoch+1}, beta_ih: {beta_ih}")

    theta_ih = calculate_theta(X, globals()['b_01'], globals()['T_01'],Zo_h)
    print(f"Epoch {epoch+1}, thetha_ih: {theta_ih}")


    sum_Delta_T_output = np.sum(delta_threshold_ho, axis=1, keepdims=True)
    delta_threshold_ih = beta_ih * sum_Delta_T_output
    print(f"Epoch {epoch+1}, delta_threshold_ih: {delta_threshold_ih}")

    delta_bias_ih = -(theta_ih)*sum_Delta_T_output
    print(f"Epoch {epoch+1}, delta_bias_ih: {delta_bias_ih}")


    # Update thresholds and biases for hidden layer
    globals()['T_01'] -=  delta_threshold_ih.mean(axis=0)
    print(f"Epoch {epoch+1}, T_01: {T_01}")

    globals()['b_01'] -=  delta_bias_ih.mean(axis=0)
    print(f"Epoch {epoch+1}, b_01: {b_01}")


    print(f'Epoch {epoch + 1}/{epochs}, Loss: {E}')

# # After training, predict using the final model parameters
# Y_pred1,Z1 = forward_propagation(X, globals()['b_01'], globals()['T_01'], n_hidden_nodes,n_input_nodes)
# Y_pred2 = relu(Y_pred1)  # Activation for hidden layer
Y_pred3, Z2 = forward_propagation(X, globals()['b_01'], globals()['T_01'], n_output_nodes, n_input_nodes)
Y_pred4 = relu(Y_pred3)  # Activation for output layer

# Assuming the prediction and actual values are both binary for classification
predicted_classes = Y_pred4 > 0.5
true_classes = y > 0.5
print("true_classes",true_classes)
accuracy = np.mean(predicted_classes == true_classes)
print(f'Model accuracy: {accuracy*100}%')
