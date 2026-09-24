#ifndef MATERIAL_TYPES_INC
#define MATERIAL_TYPES_INC

// Single source of truth for material type IDs shared across C++ and GLSL.
// Kept in numeric lockstep so CPU flatten and GPU kernel never drift.

#define MAT_SOLID     0
#define MAT_RESERVED  1  // fuzz-era deleted; do not renumber
#define MAT_GLASS     2
#define MAT_EMIT      3
#define MAT_CHECKER   4
#define MAT_IMAGE     5
#define MAT_FOG       6
#define MAT_CONDUCTOR 7
#define MAT_HET       8
#define MAT_NOISE     9
#define MAT_ANISO     10
#define MAT_DISNEY    11

#endif // MATERIAL_TYPES_INC
