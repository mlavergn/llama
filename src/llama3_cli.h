//
//  llama3_cli.h
//  llama3
//
//  Created by Marc Lavergne on 6/14/24.
//

#ifndef llama3_cli_h
#define llama3_cli_h

#import "llama3.h"

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface IXLlama3: NSObject

-(NSInteger)run:(NSArray<NSString *> *)args;

@end

NS_ASSUME_NONNULL_END

#endif /* llama3_cli_h */
