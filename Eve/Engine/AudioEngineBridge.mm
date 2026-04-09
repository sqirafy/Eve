#import "AudioEngineBridge.h"
#include "include/AudioEngine.hpp"
#include <memory>

@implementation AudioEngineBridge {
    std::unique_ptr<eve::AudioEngine> _engine;
}

- (instancetype)init {
    self = [super init];
    if (self) _engine = std::make_unique<eve::AudioEngine>();
    return self;
}

- (BOOL)loadModelAtURL:(NSURL *)modelURL {
    if (!_engine) return NO;
    return _engine->loadModel(modelURL.path.UTF8String) ? YES : NO;
}

- (BOOL)startWithMicDeviceID:(uint32_t)micDeviceID {
    if (!_engine) return NO;
    return _engine->start(micDeviceID) ? YES : NO;
}

- (void)stop {
    if (_engine) _engine->stop();
}

- (void)setPassthrough:(BOOL)passthrough {
    if (_engine) _engine->setPassthrough(passthrough);
}

- (BOOL)isRunning {
    return _engine ? _engine->isRunning() : NO;
}

- (BOOL)isBlackHoleConnected {
    return _engine ? _engine->isBlackHoleConnected() : NO;
}

@end
