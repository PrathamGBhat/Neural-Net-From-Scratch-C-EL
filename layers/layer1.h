#ifndef LAYER1_H
#define LAYER1_H

#include "dataset.h"

#define HIDDEN1_ROWS 8
#define HIDDEN1_COLS INPUT_SIZE

typedef struct {
    float weights[HIDDEN1_ROWS][HIDDEN1_COLS];
    float biases[HIDDEN1_ROWS];
    float out[HIDDEN1_ROWS];
    float delta[HIDDEN1_ROWS];
} HiddenLayer1;

HiddenLayer1 layer1_init(void);
void layer1_forward(HiddenLayer1 *l, float input[HIDDEN1_COLS]);
void layer1_backward(HiddenLayer1 *l, void *next_weights, float *next_delta, int next_size);
void layer1_update(HiddenLayer1 *l, float input[HIDDEN1_COLS], float lr);

#endif