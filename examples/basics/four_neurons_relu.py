import numpy as np


# Define the ReLU activation function for array inputs
def relu(x):
    return np.maximum(0, x)


# Function to compute the output of a network with 4 neurons
def four_neuron_network(weights, biases, inputs):

    outputs_before_activation = []
    activated_outputs = []

    for i, (weight, bias) in enumerate(zip(weights, biases), start=1):
        # Calculate the neuron's outputs before activation
        output_before_activation = np.dot(weight, inputs) + bias

        # Apply the ReLU activation function
        activated_output = relu(output_before_activation)

        # Add the calculation step for each neuron
        calculation_steps = [f"({w})*({x})" for w, x in zip(weight, inputs)]
        calculation_steps.append(f"({bias})")
        output_text = f"w{i} * x + b{i} = " + " + ".join(calculation_steps)
        output_text += f" = {output_before_activation}\n"

        outputs_before_activation.append(output_before_activation)
        activated_outputs.append(activated_output)

    # Count the parameters of each node: number of weights + 1 bias
    params_per_node = weights.shape[1] + 1

    # Count all the parameters of this network
    total_params = weights.shape[0] * params_per_node

    print(output_text)
    return outputs_before_activation, activated_outputs, params_per_node, total_params


# Given values for the 4-neuron network
weights = np.array(
    [
        [1, -1, 1],  # Neuron 1 weights
        [1, 1, 0],  # Neuron 2 weights
        [0, 1, 1],  # Neuron 3 weights
        [1, 0, 1],  # Neuron 4 weights
    ]
)
biases = np.array([-5, 0, 1, -2])
inputs = np.array([2, 1, 3])

# Perform the operation
outputs_before_activation, activated_outputs, params_per_node, total_params = (
    four_neuron_network(weights, biases, inputs)
)

# Print the outcomes
print("Outputs before ReLU activation:\n", outputs_before_activation)
print(
    "Outputs after ReLU activation: (negative values → 0; positive values → same values)\n",
    activated_outputs,
)
print("Parameters per node: (number of weights + 1 bias)\n", params_per_node)
print("Total parameters in the network:\n", total_params)
