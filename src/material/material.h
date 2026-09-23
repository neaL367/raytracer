#pragma once
// Modular material subsystem:
// - material_base.h: MatType enum, reflection/emission interfaces
// - lambertian.h:    ideal diffuse reflection + texture mapping
// - metal.h:         conductors (isotropic + anisotropic GGX, n/k, thin-film)
// - dielectric.h:    smooth/rough glass (Walter BTDF, Beer's law, Cauchy, thin-film)
// - diffuse_light.h: area light emitter + texture-mapped emission
// - isotropic.h:     volumetric phase function scattering

#include "material_base.h"
#include "lambertian.h"
#include "metal.h"
#include "dielectric.h"
#include "diffuse_light.h"
#include "isotropic.h"
