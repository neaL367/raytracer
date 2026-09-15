// Single translation unit for stb_image's implementation. The header is
// third-party code that does not pass our /W4 cleanly, so warnings are
// suppressed for exactly this include — our own headers stay warning-free.
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif
