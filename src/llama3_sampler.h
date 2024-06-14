//
//  llama3_sampler.h
//  llama3
//
//  Created by Marc Lavergne on 6/14/24.
//

#ifndef llama3_sampler_h
#define llama3_sampler_h

#import <stdlib.h>

// MARK: - Sampler

// Sampler takes logits and returns a sampled token.
// Sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex *probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// MARK: - Sampler

// Sampler takes logits and returns a sampled token.
// Sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

int sample_argmax(float *probabilities, int n);

int sample_mult(float *probabilities, int n, float coin);

int compare(const void *a, const void *b);

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin);

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);

void free_sampler(Sampler *sampler);

unsigned int random_u32(unsigned long long *state);

float random_f32(unsigned long long *state);

int sample(Sampler *sampler, float *logits);

#endif /* llama3_sampler_h */
