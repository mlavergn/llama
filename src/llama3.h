//
//  Llama3.h
//  llama3
//
//  Created by Marc Lavergne on 6/13/24.
//

#import <stdlib.h>

#import "llama3_encoder.h"
#import "llama3_nnet.h"
#import "llama3_sampler.h"
#import "llama3_transformer.h"

// MARK: - Utilities

/// return time in milliseconds, for benchmarking the model speed
long time_in_ms(void);

// MARK: - Generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps);

void read_stdin(const char *guide, char *buffer, int bufsize);

// MARK: - Chat loop

// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *cli_system_prompt, int steps);
