#include "output_layer.h"
#include "../utils/utils.h"

OutputLayer output_init(void) {
    OutputLayer l;

    // Initialize random weights, random biases, output as 0, delta as 0
    for (int j = 0; j < OUTPUT_COLS; j++) {
        l.weights[0][j] = randf();
    }
    l.biases[0] = randf();
    l.out[0]    = 0.0f;
    l.delta[0]  = 0.0f;

    return l;
}

void output_forward(OutputLayer *l, float input[OUTPUT_COLS]) {

    // Add bias to sum
    float sum = l->biases[0];

    // Calculate output from second last layer's 4 neurons
    for (int j = 0; j < OUTPUT_COLS; j++)
        sum += l->weights[0][j] * input[j];

    // Pass sum through sigmoid
    l->out[0] = sigmoid(sum);
}

void output_backward(OutputLayer *l, float target) {

    // Calculate derivative of loss function as err
    float err = target - l->out[0];

    // Calculate and store gradient in delta layer for previous layer to access
    l->delta[0] = err * sigmoid_deriv(l->out[0]);
}

void output_update(OutputLayer *l, float input[OUTPUT_COLS], float lr) {

    // Iterate through the weights of each incoming neuron from second last layer
    for (int j = 0; j < OUTPUT_COLS; j++)

        // Update each weight by stochastic gradient descent
        l->weights[0][j] += lr * l->delta[0] * input[j];
    
    // Update the bias of the last neuron
    l->biases[0] += lr * l->delta[0];
}