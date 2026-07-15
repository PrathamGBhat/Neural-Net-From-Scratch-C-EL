#include "utils.h"

float randf(void) {

    // Return random floating point value between 0 and 1
    return (((float)rand() / (float)RAND_MAX) - 0.5f) * 2.0f;
}

float sigmoid(float x) {

    // Return the output when input x is put through sigmoid
    if (x >= 0.0f)
        return 1.0f / (1.0f + expf(-x));
    float e = expf(x); // Handled positive and negative input separately to avoid overflow of floating points
    return e / (1.0f + e);
}

float sigmoid_deriv(float y) {

    // Return the output when input y is put through derivative of sigmoid
    return y * (1.0f - y);
}