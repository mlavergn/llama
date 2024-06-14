//
//  llama3_encoder.h
//  llama3
//
//  Created by Marc Lavergne on 6/14/24.
//

#ifndef llama3_encoder_h
#define llama3_encoder_h

#import <stdlib.h>

// MARK: - Byte Pair Encoding (BPE) Tokenizer [strings <-> tokens]

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

// MARK: - Byte Pair Encoding (BPE) Tokenizer [strings <-> tokens]

int compare_tokens(const void *a, const void *b);

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size);
  
void free_tokenizer(Tokenizer *t);

char *decode(Tokenizer *t, int prev_token, int token);

void safe_printf(char *piece);

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);

#endif /* llama3_encoder_h */
