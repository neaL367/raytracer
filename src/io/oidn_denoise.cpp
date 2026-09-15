// OIDN call site, isolated so main.cpp never sees the third-party API.
// Without RAYTRACER_OIDN this is a stub: same signature, always false,
// so --oidn degrades to bilateral with a note instead of a build flag maze.
#include "oidn_denoise.h"

#include <cstdio>

#ifdef RAYTRACER_OIDN
#include <OpenImageDenoise/oidn.h>

bool oidn_denoise_rt(float *color, const float *albedo, const float *normal, int width, int height,
                     bool hdr)
{
    // Explicit CPU: DEFAULT probes GPU runtimes first, which is both slower
    // to initialize and fragile on mixed systems. This renderer is CPU-fed;
    // keep the denoise where the data already lives.
    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
    oidnCommitDevice(device);
    OIDNFilter filter = oidnNewFilter(device, "RT");
    // Shared images: zero-copy views over the caller's buffers, which must
    // outlive the execute call (they do — stack vectors in main).
    oidnSetSharedFilterImage(filter, "color", color, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "albedo", const_cast<float *>(albedo), OIDN_FORMAT_FLOAT3,
                             width, height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "normal", const_cast<float *>(normal), OIDN_FORMAT_FLOAT3,
                             width, height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "output", color, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterBool(filter, "hdr", hdr);
    oidnSetFilterBool(filter, "cleanAux", true); // guides use full sampling: treat as noise-free
    oidnSetFilterInt(filter, "maxMemoryMB", 2048);
    oidnCommitFilter(filter);
    oidnExecuteFilter(filter);
    const char *message = nullptr;
    bool ok = oidnGetDeviceError(device, &message) == OIDN_ERROR_NONE;
    if (!ok)
        std::fprintf(stderr, "oidn: %s\n", message ? message : "unknown error");
    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);
    return ok;
}

#else

bool oidn_denoise_rt(float *, const float *, const float *, int, int, bool)
{
    std::fprintf(stderr, "oidn: not compiled in (configure with -DRAYTRACER_OIDN=ON)\n");
    return false;
}

#endif
