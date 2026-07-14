#include "layer2.h"
#include "../utils/utils.h"

// Initialize random weights and biases for layer 2
HiddenLayer2 layer2_init(void) {
    HiddenLayer2 l;
    for (int i = 0; i < HIDDEN2_ROWS; i++) {
        for (int j = 0; j < HIDDEN2_COLS; j++)
            l.weights[i][j] = randf();
        l.biases[i] = randf();
        l.out[i]    = 0.0f;
        l.delta[i]  = 0.0f;
    }
    return l;
}

// Activate layer 2 neurons in forward prop
void layer2_forward(HiddenLayer2 *l, float input[HIDDEN2_COLS]) {
    for (int i = 0; i < HIDDEN2_ROWS; i++) {
        float sum = l->biases[i];
        for (int j = 0; j < HIDDEN2_COLS; j++)
            sum += l->weights[i][j] * input[j];
        l->out[i] = sigmoid(sum);
    }
}

// Activate layer 2 neurons in backward prop
void layer2_backward(HiddenLayer2 *l, void *next_weights, float *next_delta, int next_size) {
    float (*w)[HIDDEN2_ROWS] = (float (*)[HIDDEN2_ROWS])next_weights;
    for (int j = 0; j < HIDDEN2_ROWS; j++) {
        float err = 0.0f;
        for (int i = 0; i < next_size; i++)
            err += w[i][j] * next_delta[i];
        l->delta[j] = err * sigmoid_deriv(l->out[j]);
    }
}

// Update weights in layer 2
void layer2_update(HiddenLayer2 *l, float input[HIDDEN2_COLS], float lr) {
    for (int i = 0; i < HIDDEN2_ROWS; i++) {
        for (int j = 0; j < HIDDEN2_COLS; j++)
            l->weights[i][j] += lr * l->delta[i] * input[j];
        l->biases[i] += lr * l->delta[i];
    }
}