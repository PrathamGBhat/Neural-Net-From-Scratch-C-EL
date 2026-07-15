#include "layer1.h"
#include "../utils/utils.h"

HiddenLayer1 layer1_init(void) {
    HiddenLayer1 l;

    // Initialize random weights, random biases, output as 0, delta as 0
    for (int i = 0; i < HIDDEN1_ROWS; i++) {
        for (int j = 0; j < HIDDEN1_COLS; j++)
            l.weights[i][j] = randf();
        l.biases[i] = randf();
        l.out[i]    = 0.0f;
        l.delta[i]  = 0.0f;
    }

    return l;
}

void layer1_forward(HiddenLayer1 *l, float input[HIDDEN1_COLS]) {

    // Iterate through each neuron
    for (int i = 0; i < HIDDEN1_ROWS; i++) {
        
        // Add the bias to sum
        float sum = l->biases[i];

        // Add the product of all weights with input to sum
        for (int j = 0; j < HIDDEN1_COLS; j++)
            sum += l->weights[i][j] * input[j];

        // Pass sum through activation function
        l->out[i] = sigmoid(sum);
    }
}

void layer1_backward(HiddenLayer1 *l, void *next_weights, float *next_delta, int next_size) {

    // Setup float type pointer to point to the outgoing weights of the current layer
    float (*w)[HIDDEN1_ROWS] = (float (*)[HIDDEN1_ROWS])next_weights;

    // Iterate through each neuron
    for (int j = 0; j < HIDDEN1_ROWS; j++) {
        float err = 0.0f;

        // Calculate contribution of current neuron to next layers neurons - sum of weight x next gradient for all the outgoing weights
        for (int i = 0; i < next_size; i++)
            err += w[i][j] * next_delta[i];

        // Assign gradient of each neuron for previous layer to access
        l->delta[j] = err * sigmoid_deriv(l->out[j]);
    }
}

void layer1_update(HiddenLayer1 *l, float input[HIDDEN1_COLS], float lr) {

    // Iterate through each neuron
    for (int i = 0; i < HIDDEN1_ROWS; i++) {
        
        // Update weights by stochastic gradient descent
        for (int j = 0; j < HIDDEN1_COLS; j++)
            l->weights[i][j] += lr * l->delta[i] * input[j];

        // Update biases by stochastic gradient descent
        l->biases[i] += lr * l->delta[i];
    }
}