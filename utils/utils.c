#include "utils.h"

float randf(void) {
    return (((float)rand() / (float)RAND_MAX) - 0.5f) * 2.0f;
}
float sigmoid(float x) {
    if (x >= 0.0f)
        return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

float sigmoid_deriv(float y) {
    return y * (1.0f - y);
}