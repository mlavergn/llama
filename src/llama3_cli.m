//
//  Llam3.m
//  llama3
//
//  Created by Marc Lavergne on 6/13/24.
//

#import "llama3_cli.h"

// MARK: - CLI

void error_usage(void) {
    fprintf(stderr, "Usage:  run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 4096 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "    -t <float> temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "    -p <float> p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "    -s <int> random seed, default time(NULL)\n");
    fprintf(stderr, "    -n <int> number of steps to run for, default 4096. 0 = max_seq_len\n");
    fprintf(stderr, "    -i <string> input prompt\n");
    fprintf(stderr, "    -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "    -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "    -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int run(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = NULL;    // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 4096;                // number of steps to run for
    char *prompt = NULL;             // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";         // generate|chat
    char *system_prompt = NULL;      // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) {
        checkpoint_path = argv[1];
    } else {
        error_usage();
    }
    for (int i = 2; i < argc; i += 2) {
        // do some basic validation
        if (i + 1 >= argc) {
            error_usage();
        } // must have arg after flag
        if (argv[i][0] != '-') {
            error_usage();
        } // must start with dash
        if (strlen(argv[i]) != 2) {
            error_usage();
        } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') {
            temperature = atof(argv[i + 1]);
        } else if (argv[i][1] == 'p') {
            topp = atof(argv[i + 1]);
        } else if (argv[i][1] == 's') {
            rng_seed = atoi(argv[i + 1]);
        } else if (argv[i][1] == 'n') {
            steps = atoi(argv[i + 1]);
        } else if (argv[i][1] == 'i') {
            prompt = argv[i + 1];
        } else if (argv[i][1] == 'z') {
            tokenizer_path = argv[i + 1];
        } else if (argv[i][1] == 'm') {
            mode = argv[i + 1];
        } else if (argv[i][1] == 'y') {
            system_prompt = argv[i + 1];
        } else {
            error_usage();
        }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) {
        rng_seed = (unsigned int)time(NULL);
    }
    if (temperature < 0.0) {
        temperature = 0.0;
    }
    if (topp < 0.0 || 1.0 < topp) {
        topp = 0.9;
    }
    if (steps < 0) {
        steps = 0;
    }

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}

@implementation IXLlama3

-(NSInteger)run:(NSArray<NSString *> *)args {
    NSInteger argc = args.count;
    char *argv[argc];
    for (NSInteger i=0; i<argc; i++) {
        argv[i] = (char *)[args[i] cStringUsingEncoding:NSUTF8StringEncoding];
    }
    int rc = run((int)argc, argv);
    return (NSInteger)rc;
}

@end
