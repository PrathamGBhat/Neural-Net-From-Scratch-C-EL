#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "layers/dataset.h"
#include "layers/layer1.h"
#include "layers/layer2.h"
#include "layers/output_layer.h"

#define EPOCHS 100000
#define LR     0.5f

int main(void) {
    srand((unsigned)time(NULL));

    HiddenLayer1 l1  = layer1_init();
    HiddenLayer2 l2  = layer2_init();
    OutputLayer  out = output_init();

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        
        //trainin
        for (int s = 0; s < SAMPLES; s++) {
            float *x = inputs[s];
            float  y = labels[s];

            layer1_forward(&l1, x);
            layer2_forward(&l2, l1.out);
            output_forward(&out, l2.out);

            output_backward(&out, y);
            layer2_backward(&l2, out.weights, out.delta, OUTPUT_ROWS); 
            layer1_backward(&l1, l2.weights,  l2.delta,  HIDDEN2_ROWS);

            output_update(&out, l2.out, LR);
            layer2_update(&l2,  l1.out, LR);
            layer1_update(&l1,  x,      LR);
        }

        // evaluation
        float total_loss = 0.0f;
        int   correct    = 0;

        for (int s = 0; s < SAMPLES; s++) {
            float *x = inputs[s];
            float  y = labels[s];

            
            layer1_forward(&l1, x);
            layer2_forward(&l2, l1.out);
            output_forward(&out, l2.out);

            float err = y - out.out[0];
            total_loss += 0.5f * err * err;
            
            // check accuracy based on standard 0.5 threshold
            if ((out.out[0] >= 0.5f && y == 1.0f) || (out.out[0] < 0.5f && y == 0.0f)) {
                correct++;
            }
        }

        
        if (epoch % 500 == 0) {
            printf("Epoch %d  Loss: %.6f  Accuracy: %.1f%%\n",
                   epoch, total_loss / SAMPLES,
                   (float)correct / SAMPLES * 100.0f);
        }
    }

    printf("\nFinal Results:\n");
    for (int s = 0; s < SAMPLES; s++) {
        float *x = inputs[s];
        float  y = labels[s];
        layer1_forward(&l1, x);
        layer2_forward(&l2, l1.out);
        output_forward(&out, l2.out);
        printf("Input: [%.0f, %.0f, %.0f, %.0f] Expected: %.0f Actual: %.4f %s\n",
               x[0], x[1], x[2], x[3], y, out.out[0],
               ((out.out[0] >= 0.5f && y == 1.0f) || (out.out[0] < 0.5f && y == 0.0f)) ? "OK" : "FAIL");
    }
}