import numpy as np


# Define the ReLU activation function for scalar inputs
def relu(x):
    return np.maximum(0, x)


# Function to compute the output of a single neuron network
def single_neuron_network(weights, bias, inputs):
    # Calculate the neuron's output before activation
    output_before_activation = sum(w * x for w, x in zip(weights, inputs)) + bias

    # Apply the ReLU activation function
    activated_output = relu(output_before_activation)

    # Add the calculation step
    calculation_steps = [f"({w})*({x})" for w, x in zip(weights, inputs)]
    calculation_steps.append(f"({bias})")
    output_text = "w*x + b = " + " + ".join(calculation_steps)
    output_text += f" = {output_before_activation}\n"

    # Add the ReLU step
    output_text += f"ReLU: {output_before_activation} → {activated_output} because "
    output_text += f"{output_before_activation} is {'negative' if output_before_activation < 0 else 'non-negative'}."

    print(output_text)
    return output_before_activation, activated_output


# Given values for single node network
inputs = np.array([2, 1, 3])
weights = np.array([1, -1, 1])
bias = -5  # bias should be a scalar, not an array

# Perform the operation
output_before_activation, activated_output = single_neuron_network(
    weights, bias, inputs
)
