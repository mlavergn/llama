//
// llama3.c
// llama3
//
// Created by Marc Lavergne on 6/13/24.
//

/* Inference for Llama-3 Transformer model in pure C */

#import "llama3.h"

#import <ctype.h>
#import <fcntl.h>
#import <math.h>
#import <stdio.h>
#import <string.h>
#import <sys/mman.h>
#import <time.h>
#import <unistd.h>

// MARK: - Utilities

/// return time in milliseconds, for benchmarking the model speed
long time_in_ms(void) {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// MARK: - Generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                            char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;        // used to time our code, only initialized after first iteration
    int next; // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                                    // position in the sequence

    while (pos < steps) {
        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt
            // token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits
        // sequences
        if ((next == 128001 || next == 128009) && pos > num_prompt_tokens)
            break;
        // print the token as string, decode it with the Tokenizer object
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) {
            start = time_in_ms();
        }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first
    // iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n",
                        (pos - 1) / (double)(end - start) * 1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, int bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// MARK: - Chat loop

// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are somewhat haphazardly and unsafely set atm
    char *system_prompt = (char *)malloc(32768 * sizeof(char));
    char *user_prompt = (char *)malloc(32768 * sizeof(char));
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
    int *system_prompt_tokens = (int *)malloc(32768 * sizeof(int));
    int *user_prompt_tokens = (int *)malloc(32768 * sizeof(int));
    int user_idx = 0;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next = 0;         // will store the next token in the sequence
    int token;            // stores the current token to feed into the transformer

    int pos = 0; // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                prompt_tokens[num_prompt_tokens++] = 128000; // "<|begin_of_text|>"
                prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
                prompt_tokens[num_prompt_tokens++] = 9125;   // "system"
                prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
                prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, 32768);
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
                if (system_prompt != NULL) {
                    int num_system_prompt_tokens = 0;
                    encode(tokenizer, system_prompt, 0, 0, system_prompt_tokens,
                                 &num_system_prompt_tokens);
                    for (int i = 0; i < num_system_prompt_tokens; i++) {
                        prompt_tokens[num_prompt_tokens++] = system_prompt_tokens[i];
                    }
                }
                prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
            } else {
                num_prompt_tokens = 0;
            }
            prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
            prompt_tokens[num_prompt_tokens++] = 882;    // "user"
            prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
            prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User (or exit): ", user_prompt, 32768);
                if (strcmp(user_prompt, "exit") == 0)
                    break;
            }
            int num_user_prompt_tokens = 0;
            // encode the user prompt into tokens
            encode(tokenizer, user_prompt, 0, 0, user_prompt_tokens,
                         &num_user_prompt_tokens);
            for (int i = 0; i < num_user_prompt_tokens; i++) {
                prompt_tokens[num_prompt_tokens++] = user_prompt_tokens[i];
            }
            prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
            prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
            prompt_tokens[num_prompt_tokens++] = 78191;  // "assistant"
            prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
            prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"

            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt
            // token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=128009) token ends the Assistant turn
        if (user_idx >= num_prompt_tokens && (token == 128009 || token == 128001)) {
            user_turn = 1;
        }

        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 128009 && next != 128001 &&
                next != 128006) {
            // the Assistant is responding, so print its output
            char *piece = decode(tokenizer, token, next);
            safe_printf(
                    piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (user_idx >= num_prompt_tokens && (next == 128009 || next == 128001)) {
            printf("\n");
        }
    }
    printf("\n");
    free(prompt_tokens);
    free(system_prompt_tokens);
    free(user_prompt_tokens);
    free(system_prompt);
    free(user_prompt);
}
