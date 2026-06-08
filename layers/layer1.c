#include "layer1.h"
#include "../utils/utils.h"

// Initialize random weights and biases for layer 1
HiddenLayer1 layer1_init(void) {
    HiddenLayer1 l;
    for (int i = 0; i < HIDDEN1_ROWS; i++) {
        for (int j = 0; j < HIDDEN1_COLS; j++)
            l.weights[i][j] = randf();
        l.biases[i] = randf();
        l.out[i]    = 0.0f;
        l.delta[i]  = 0.0f;
    }
    return l;
}

// Activate layer 1 neurons in forward prop
void layer1_forward(HiddenLayer1 *l, float input[HIDDEN1_COLS]) {
    for (int i = 0; i < HIDDEN1_ROWS; i++) {
        float sum = l->biases[i];
        for (int j = 0; j < HIDDEN1_COLS; j++)
            sum += l->weights[i][j] * input[j];
        l->out[i] = sigmoid(sum);
    }
}

// Activate layer 1 neurons in backward prop
void layer1_backward(HiddenLayer1 *l, void *next_weights, float *next_delta, int next_size) {
    float (*w)[HIDDEN1_ROWS] = (float (*)[HIDDEN1_ROWS])next_weights;
    for (int j = 0; j < HIDDEN1_ROWS; j++) {
        float err = 0.0f;
        for (int i = 0; i < next_size; i++)
            err += w[i][j] * next_delta[i];
        l->delta[j] = err * sigmoid_deriv(l->out[j]);
    }
}

// Update weights in layer 1
void layer1_update(HiddenLayer1 *l, float input[HIDDEN1_COLS], float lr) {
    for (int i = 0; i < HIDDEN1_ROWS; i++) {
        for (int j = 0; j < HIDDEN1_COLS; j++)
            l->weights[i][j] += lr * l->delta[i] * input[j];
        l->biases[i] += lr * l->delta[i];
    }
}