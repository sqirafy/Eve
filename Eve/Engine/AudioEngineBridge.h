#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface AudioEngineBridge : NSObject

- (BOOL)loadModelAtURL:(NSURL *)modelURL;
- (BOOL)startWithMicDeviceID:(uint32_t)micDeviceID;
- (void)stop;
- (void)setPassthrough:(BOOL)passthrough;

@property (nonatomic, readonly) BOOL isRunning;
@property (nonatomic, readonly) BOOL isBlackHoleConnected;

@end

NS_ASSUME_NONNULL_END
