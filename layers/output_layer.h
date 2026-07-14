#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

#include "layer2.h"

#define OUTPUT_ROWS 1
#define OUTPUT_COLS HIDDEN2_ROWS

typedef struct {
    float weights[OUTPUT_ROWS][OUTPUT_COLS];
    float biases[OUTPUT_ROWS];
    float out[OUTPUT_ROWS];
    float delta[OUTPUT_ROWS];
} OutputLayer;

OutputLayer output_init(void);
void output_forward(OutputLayer *l, float input[OUTPUT_COLS]);
void output_backward(OutputLayer *l, float target);
void output_update(OutputLayer *l, float input[OUTPUT_COLS], float lr);

#endif