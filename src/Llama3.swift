//
//  Llama3.swift
//  llama3
//
//  Created by Marc Lavergne on 7/20/24.
//

//import IXLlama3

/*
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
 */

struct Llama3Request {
    let temperature: Float = 1.0
    let nucleusSampling: Float = 0.9
    let randomSeed: Int = Int(Date().timeIntervalSince1970)
    let steps: Int = 4096
    let input: String = "Once upon a time"
    let tokenizerPath: String?
    let mode: String = "generate"
    let prompt: String?
}

class Llama3 {
    func handle(_ req: Llama3Request) -> Int {
        var args: [String] = ["run", "model.bin"]
        
        args.append("-n")
        args.append("\(req.steps)")
        args.append("-i")
        args.append(req.input)

        let ix = IXLlama3()
        return ix.run(args)
    }
}
