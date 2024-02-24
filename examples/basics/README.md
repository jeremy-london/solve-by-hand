# Deep Learning and Neural Network Basics

## Exercise 1

### [Matrix Multiplication](https://lnkd.in/gXKnQQF3)

In this exercise, we delve into the critical technique of matrix multiplication, a cornerstone of neural network operations. By focusing on the multiplication of matrices A and B to produce matrix C, students gain insights into the practical aspects of linear algebra essential for deep learning. The exercise illuminates the relationship between the dimensions of the matrices involved, offering a clear view of how the size of one matrix influences the others.

- Grasp C's dimensions from A's rows and B's columns.
- Scaling A, B, or C alters others' dimensions.
- Combine A's row and B's column vectors via dot product.
- Operations' compactness enables neural network architecture representation.

Code: [matrix_multiplication.py](./matrix_multiplication.py)

## Exercise 2

### [Single Neuron Network](https://lnkd.in/gKqEGPgf)

This exercise introduces the concept of a single neuron, which is the basic building block of neural networks. You will learn how to calculate the output of a neuron given an input vector and a set of weights.

1. `w * x + b = (1)*(2) + (-1)*(1) + (1)*(3) + (-5) = -1`
2. ReLU: -1 → 0 because -1 is negative.

Code: [single_neuron_relu.py](./single_neuron_relu.py)

## Exercise 3

### [Four Neurons](https://lnkd.in/gc-qwJ6X)

Expanding from a single neuron, this exercise explores a layer with four neurons. You can practice calculating the outputs of each neuron within a layer, using matrix operations and the ReLU activation function.

1. Multiply the same input x with each neuron’s weights plus a bias term in each row

```
w1 * x + b1 = 2 - 1 + 3 - 5 = -1
w2 * x + b2 = 2 + 1 = 3
w3 * x + b3 = 1 + 3 + 1 = 5
w4 * x + b4 = 2 + 3 - 2 = 3
```

2. ReLU: negative values → 0; positive values → same values
3. Count the parameters of each node: `3 + 1`
4. Count all the parameters of this network: `4 * (3 + 1) = 16`

Code: [four_neurons_relu.py](./four_neurons_relu.py)

## Exercise 4

### [Hidden Layer](https://lnkd.in/gwDXEGsM)

Here, we are introduced to the concept of hidden layers. The exercise walks through the process of adding a hidden layer to a neural network and the computations involved.

1. Hidden Layer: `W1 * x + b1 → ReLU → h`
2. Output Layer:  `W2 * h + b2 → ReLU → y`
3. See the correspondence between weights and nodes

```
# of weights in a hidden node = # of nodes in the input layer
# of weights in an output node = # of nodes in the hidden layer
```

4. Count all the parameters of this network: `4 * (3 + 1) + 2 * (4 + 1) = 16 + 10 = 26`

Code: [hidden_layer_relu.py](./hidden_layer_relu.py)

## Exercise 5

### [Batch of Three Inputs](https://lnkd.in/eMjpAZta)

This exercise demonstrates how a neural network handles a batch of inputs. This exercise walks uses three different input vectors, and helps in understanding how weights are applied across a batch.

1. Inputs: The network takes a batch of three input vectors `(x1, x2, x3)`, each corresponds to a column in the top-most matrix.
2. Shade each corresponding pair of weight vector (row) and feature vector (column) using a unique color
3. Calculate each missing value by taking the dot product between corresponding weight vector and feature vector
4. See that the same set of weights are applied to each input `(x1, x2, x3)`

Code: [three_input_batch_relu.py](./three_input_batch_relu.py)

## Exercise 6

### [Seven Layers, aka. Multi-Layer Perceptron (MLP)](https://lnkd.in/gjJRxPsv)

Building upon the previous exercises, you can now explore a multi-layer perceptron with several layers. The exercise covers the forward propagation through multiple layers and the activation functions applied at each layer.

This exercise uses a compact form, making two simplifying assumptions:

1. All biases are zeros.
2. ReLU is directly applied to each cell (except for the output layer), for example, crossing out -5 and replacing it with 0.

Using this compact form, we can easily stack many layers to form a deeper network, like this seven layer network.

This exercise allows students to practice the following:

1. Inputs: The network takes a batch of two input vectors `(x1, x2)`
2. Layers: The network has seven layers.
3. Draw links between two layers of nodes
4. Shade corresponding weights (left) and links (right) in matching colors
5. Apply ReLU to "deactivate" negative values to 0 (except for the output layer).
6. Calculate one missing value in each layer

```
Layer 1: 0 x 3 + 1 x 4 + 1 x 5 = 9
Layer 2: 0 x 5 + 0 x 4 + 1 x 3 + 1 x 7 + (-1) x 9 = 1
Layer 3: 1 x 6 + 2 x 1 = 8
Layer 4: 0 x 7 + (-1) x 5 + 1 x 8 = 3
Layer 5: 1 x 2 + 0 x 3 = 2
```

Code: [multi_layer_perceptron_relu.py](./multi_layer_perceptron_relu.py)

## Exercise 7

### [Backpropagation](https://lnkd.in/gsiU2uc2)

Diving deeper into the intricacies of neural networks, this example takes you through the journey of backpropagation in a multi-layer perceptron, encompassing three distinct layers. The focus here is on mastering the art of navigating backwards through the network—starting from the output layer and meticulously calculating the gradients at each step. This reverse engineering process is pivotal for optimizing the network's weights and biases, ultimately refining its ability to make accurate predictions. This backpropagation exercise is designed to provide a comprehensive grasp of the underlying mechanics that empower neural networks to learn from data.

In the context of the code and the backpropagation process, the letter "d" in the formulas (such as dL/dW1, dL/db1, etc.) represents the partial derivative of the loss function with respect to the variable that follows. Essentially, it's a notation that comes from calculus, where "d" stands for "derivative." Specifically:

- `dL`: The "change" in the loss function, L.
- `dW1`, `dW2`, `dW3`: The "change" in the weights of layers 1, 2, and 3, respectively.
- `db1`, `db2`, `db3`: The "change" in the biases of layers 1, 2, and 3, respectively.
- `da1`, `da2`: The "change" in the activations of layers 1 and 2, respectively.
- `dz1`, `dz2`, `dz3`: The "change" in the pre-activation (linear transformation before the activation function) of layers 1, 2, and 3, respectively.

>When you see  `dL/dW1`, it's read as "the gradient of the loss with respect to the weights of layer 1." This tells us how much the loss function is expected to change with a small change in the weights of layer 1.

 1. 𝗙𝗼𝗿𝘄𝗮𝗿𝗱 𝗣𝗮𝘀𝘀: Given a multi layer perceptron (3 levels), an input vector X, predictions `Y^{Pred} = [0.5, 0.5, 0]`, and ground truth label `Y^{Target} = [0, 1, 0]`.
 2. 𝗕𝗮𝗰𝗸𝗽𝗿𝗼𝗽𝗮𝗴𝗮𝘁𝗶𝗼𝗻: Insert cells to hold our calculations.
 3. 𝗟𝗮𝘆𝗲𝗿 𝟯 - 𝗦𝗼𝗳𝘁𝗺𝗮𝘅 (blue): Calculate `∂L / ∂z3` directly using the simple equation: `Y^{Pred} - Y^{Target} = [0.5, -0.5, 0]`. This simple equation is the benefit of using Softmax and Cross Entropy Loss together.
 4. 𝗟𝗮𝘆𝗲𝗿 𝟯 - 𝗪𝗲𝗶𝗴𝗵𝘁𝘀 & 𝗕𝗶𝗮𝘀𝗲𝘀: Calculate `∂L / ∂W3` and `∂L / ∂b3` by multiplying `∂L / ∂z3` and `[ a2 | 1 ]`.
 5. 𝗟𝗮𝘆𝗲𝗿 𝟮 - 𝗔𝗰𝘁𝗶𝘃𝗮𝘁𝗶𝗼𝗻𝘀: Calculate `∂L / ∂a2` by multiplying `∂L / ∂z3` and `W3`.
 6. 𝗟𝗮𝘆𝗲𝗿 𝟮 - 𝗥𝗲𝗟𝗨: Calculate `∂L / ∂z2` by multiplying `∂L / ∂a2` with `1` for positive values and `0` otherwise.
 7. 𝗟𝗮𝘆𝗲𝗿 𝟮 - 𝗪𝗲𝗶𝗴𝗵𝘁𝘀 & 𝗕𝗶𝗮𝘀𝗲𝘀: Calculate `∂L / ∂W2` and `∂L / ∂b2` by multiplying `∂L / ∂z2` and `[ a1 | 1 ]`.
 8. 𝗟𝗮𝘆𝗲𝗿 𝟭 - 𝗔𝗰𝘁𝗶𝘃𝗮𝘁𝗶𝗼𝗻𝘀: Calculate `∂L / ∂a1` by multiplying `∂L / ∂z2` and `W2`.
 9. 𝗟𝗮𝘆𝗲𝗿 𝟭 - 𝗥𝗲𝗟𝗨 (blue): Calculate `∂L / ∂z1` by multiplying `∂L / ∂a1` with `1` for positive values and `0` otherwise.
10. 𝗟𝗮𝘆𝗲𝗿 𝟭 - 𝗪𝗲𝗶𝗴𝗵𝘁𝘀 & 𝗕𝗶𝗮𝘀𝗲𝘀: Calculate `∂L / ∂W1` and `∂L / ∂b1` by multiplying `∂L / ∂z1` and `[ x | 1 ]`.
11. 𝗚𝗿𝗮𝗱𝗶𝗲𝗻𝘁 𝗗𝗲𝘀𝗰𝗲𝗻𝘁: Update weights and biases (typically a learning rate is applied here).

Code: [backpropagation.py](./backpropagation.py)
