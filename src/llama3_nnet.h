//
//  llama3_nnet.h
//  llama3
//
//  Created by Marc Lavergne on 6/14/24.
//

#ifndef llama3_nnet_h
#define llama3_nnet_h

#import "llama3_transformer.h"

// MARK: - Neural Network Blocks

void rmsnorm(float *o, float *x, float *weight, int size);

void softmax(float *x, int size);

// PERFORMANCE CRITICAL
void matmul(float *xout, float *x, float *w, int n, int d);

float *forward(Transformer *transformer, int token, int pos);

#endif /* llama3_nnet_h */
