#include "output_layer.h"
#include "../utils/utils.h"

OutputLayer output_init(void) {
    OutputLayer l;
    for (int j = 0; j < OUTPUT_COLS; j++) {
        l.weights[0][j] = randf();
    }
    l.biases[0] = randf();
    l.out[0]    = 0.0f;
    l.delta[0]  = 0.0f;
    return l;
}

void output_forward(OutputLayer *l, float input[OUTPUT_COLS]) {
    float sum = l->biases[0];
    for (int j = 0; j < OUTPUT_COLS; j++)
        sum += l->weights[0][j] * input[j];
    l->out[0] = sigmoid(sum);
}

void output_backward(OutputLayer *l, float target) {
    float err = target - l->out[0];
    l->delta[0] = err * sigmoid_deriv(l->out[0]);
}

void output_update(OutputLayer *l, float input[OUTPUT_COLS], float lr) {
    for (int j = 0; j < OUTPUT_COLS; j++)
        l->weights[0][j] += lr * l->delta[0] * input[j];
    l->biases[0] += lr * l->delta[0];
}