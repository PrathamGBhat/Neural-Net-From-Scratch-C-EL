#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Import all layers of the neural net
#include "layers/dataset.h"
#include "layers/layer1.h"
#include "layers/layer2.h"
#include "layers/output_layer.h"

// Define hyperparameters
#define EPOCHS 100000
#define LR     0.5f

int main(void) {

    // Initialize seed for rand() (in utils.c)
    srand((unsigned)time(NULL));

    // Initialize all layers
    HiddenLayer1 l1  = layer1_init();
    HiddenLayer2 l2  = layer2_init();
    OutputLayer  out = output_init();

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        
        // 1 pass over the dataset
        for (int s = 0; s < SAMPLES; s++) {

            // Inputs and labels
            float *x = inputs[s];
            float  y = labels[s];

            // Forward prop
            layer1_forward(&l1, x);
            layer2_forward(&l2, l1.out);
            output_forward(&out, l2.out);

            // Backward prop
            output_backward(&out, y);
            layer2_backward(&l2, out.weights, out.delta, OUTPUT_ROWS); 
            layer1_backward(&l1, l2.weights,  l2.delta,  HIDDEN2_ROWS);

            // Update parameters
            output_update(&out, l2.out, LR);
            layer2_update(&l2,  l1.out, LR);
            layer1_update(&l1,  x,      LR);
        }

        // Evaluation
        float total_loss = 0.0f;
        int correct    = 0;

        for (int s = 0; s < SAMPLES; s++) {

            // Inputs and labels
            float *x = inputs[s];
            float  y = labels[s];
            
            // Forward prop for evaluation
            layer1_forward(&l1, x);
            layer2_forward(&l2, l1.out);
            output_forward(&out, l2.out);

            // Mean Squared Error
            float err = y - out.out[0];
            total_loss += 0.5f * err * err;
            
            // Track correct predictions
            if ((out.out[0] >= 0.5f && y == 1.0f) || (out.out[0] < 0.5f && y == 0.0f)) {
                correct++;
            }
        }

        // Print results only for epochs which are multiples of 500
        if (epoch % 500 == 0) {
            printf("Epoch %d  Loss: %.6f  Accuracy: %.1f%%\n",
                   epoch, total_loss / SAMPLES,
                   (float)correct / SAMPLES * 100.0f);
        }
    }

    // Final predictions and comparisons
    printf("\nFinal Results:\n");
    for (int s = 0; s < SAMPLES; s++) {

        // Inputs and labels
        float *x = inputs[s];
        float  y = labels[s];

        // Single forward pass
        layer1_forward(&l1, x);
        layer2_forward(&l2, l1.out);
        output_forward(&out, l2.out);

        // Display results
        printf("Input: [%.0f, %.0f, %.0f, %.0f] Expected: %.0f Actual: %.4f %s\n",
               x[0], x[1], x[2], x[3], y, out.out[0],
               ((out.out[0] >= 0.5f && y == 1.0f) || (out.out[0] < 0.5f && y == 0.0f)) ? "OK" : "FAIL");
    }
}