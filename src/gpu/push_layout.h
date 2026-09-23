#pragma once
#include <cstddef>
#include <cstdint>

// Single-source layout for Vulkan push constants.
// Defined in push_layout.inc and shared between C++ and GLSL.

enum class PushWordIndex {
#define X(type, name, idx) name = idx,
#include "push_layout.inc"
#undef X
    COUNT
};

inline constexpr int kPushWordCount = static_cast<int>(PushWordIndex::COUNT);
inline constexpr size_t kPushByteSize = (size_t)kPushWordCount * sizeof(uint32_t);

#define X(type, name, idx) inline constexpr int kPush_##name = idx;
#include "push_layout.inc"
#undef X

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4201) // nameless struct/union
#endif
struct PushConstants {
    union {
        uint32_t words[kPushWordCount];
        struct {
#define X(type, name, idx) type name;
#include "push_layout.inc"
#undef X
        };
    };

    PushConstants() {
        for (int i = 0; i < kPushWordCount; ++i)
            words[i] = 0;
    }
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

static_assert(kPushWordCount == 16, "Push constant word count must be 16");
static_assert(sizeof(PushConstants) == 64, "Push constant block size must be 64 bytes");
