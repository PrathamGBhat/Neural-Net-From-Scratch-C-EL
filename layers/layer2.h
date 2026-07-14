#ifndef LAYER2_H
#define LAYER2_H

#include "layer1.h"

#define HIDDEN2_ROWS 4
#define HIDDEN2_COLS HIDDEN1_ROWS

typedef struct {
    float weights[HIDDEN2_ROWS][HIDDEN2_COLS];
    float biases[HIDDEN2_ROWS];
    float out[HIDDEN2_ROWS];
    float delta[HIDDEN2_ROWS];
} HiddenLayer2;

HiddenLayer2 layer2_init(void);
void layer2_forward(HiddenLayer2 *l, float input[HIDDEN2_COLS]);
void layer2_backward(HiddenLayer2 *l, void *next_weights, float *next_delta, int next_size);
void layer2_update(HiddenLayer2 *l, float input[HIDDEN2_COLS], float lr);

#endif