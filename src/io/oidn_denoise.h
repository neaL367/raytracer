#pragma once
// Intel OIDN path-trace denoiser (RT filter) over linear HDR + albedo +
// normal guides. Operates on float triplets, in place on color. Returns
// false when unavailable (built without OIDN, or an API failure) so the
// caller can fall back to bilateral — denoising must never lose pixels.
#include <cstddef>

bool oidn_denoise_rt(float *color, const float *albedo, const float *normal, int width, int height);
