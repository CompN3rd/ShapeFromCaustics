#pragma once

#include <optixu/optixu_vector_types.h>

struct PerRayData
{
	optix::float2 uv;
	long long     obj_ind;
	long long     tri_ind;
	float         d;
};