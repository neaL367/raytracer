#pragma once
// Intel OIDN path-trace denoiser (RT filter) over linear HDR + albedo +
// normal guides. Operates on float triplets, in place on color. Returns
// false when unavailable (built without OIDN, or an API failure) so the
// caller can fall back to bilateral — denoising must never lose pixels.
// hdr=false treats inputs as display-referred (file mode on tonemapped
// PPMs); hdr=true is linear pipeline data.
#include <cstddef>

bool oidn_denoise_rt(float *color, const float *albedo, const float *normal, int width, int height,
                     bool hdr = true);
