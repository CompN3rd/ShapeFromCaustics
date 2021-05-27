#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>
#include "prd.h"

using namespace optix;

// globals per context
rtDeclareVariable(float, scene_epsilon, , "Scene epsilon for tracing");
rtDeclareVariable(unsigned int, launch_index, rtLaunchIndex, );
rtDeclareVariable(rtObject, top_object, , );

// per ray variables
rtDeclareVariable(PerRayData, prd_ray, rtPayload, );
rtDeclareVariable(float, dist, rtIntersectionDistance, );
rtDeclareVariable(int, object_index, , "The index of hit object");

// input buffer from ray struct
rtBuffer<float3, 1> ray_origins;
rtBuffer<float3, 1> ray_directions;

// output buffers for normal rays
rtBuffer<float, 1>     d_buffer;
rtBuffer<float2, 1>    uv_buffer;  // barycentric coordinates
rtBuffer<long long, 1> object_index_buffer;
rtBuffer<long long, 1> tri_index_buffer;

// ouptut buffer for shadow rays
rtBuffer<char, 1> shadow_buffer;

// ---------------------------------------------------------------------------------
// Creates rays from given buffer objects
RT_PROGRAM void create_rays(void)
{
	// for this entry point we assume normalized directions
	Ray r(ray_origins[launch_index], ray_directions[launch_index], 0, scene_epsilon);

	PerRayData prd;
	rtTrace(top_object, r, prd);

	// copy values back to output buffers
	d_buffer[launch_index]            = prd.d;
	uv_buffer[launch_index]           = prd.uv;
	object_index_buffer[launch_index] = prd.obj_ind;
	tri_index_buffer[launch_index]    = prd.tri_ind;
}
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// Creates rays from given buffer objects
RT_PROGRAM void create_shadow_rays(void)
{
	// for this entry point we assume that the length of ray_directions refers to the valid range of the ray
	Ray r(ray_origins[launch_index],
	      normalize(ray_directions[launch_index]),
	      0,
	      scene_epsilon,
	      length(ray_directions[launch_index]) - scene_epsilon);

	PerRayData prd;
	rtTrace(top_object, r, prd);

	// copy values back to output buffers
	shadow_buffer[launch_index] = prd.obj_ind < 0 ? 1 : 0;
}
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// Creates rays from given buffer objects
RT_PROGRAM void create_infinite_shadow_rays(void)
{
	Ray r(ray_origins[launch_index], ray_directions[launch_index], 0, scene_epsilon);

	PerRayData prd;
	rtTrace(top_object, r, prd);

	// if we hit something, we have hit the light source (as our scene graph has been changed)
	// so we write back a positive result
	shadow_buffer[launch_index] = prd.obj_ind >= 0 ? 1 : 0;
}
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// Closest hit program
RT_PROGRAM void closest_hit(void)
{
	prd_ray.d       = dist;
	prd_ray.uv      = rtGetTriangleBarycentrics();
	prd_ray.obj_ind = object_index;
	prd_ray.tri_ind = rtGetPrimitiveIndex();
}
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// The miss program sets everything such that is is clear nothing is hit
RT_PROGRAM void miss(void)
{
	prd_ray.d       = -2.0f;
	prd_ray.uv      = make_float2(-1.0f, -1.0f);
	prd_ray.obj_ind = -1;
	prd_ray.tri_ind = -1;
}
// ---------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------
// Exception program
RT_PROGRAM void exception(void) { rtPrintExceptionDetails(); }
// ---------------------------------------------------------------------------------