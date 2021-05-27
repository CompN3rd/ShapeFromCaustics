#include "utils.hpp"

#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <optix_world.h>
#include <optixu/optixpp_namespace.h>

namespace py = pybind11;

// Hardcoded: get path of PyOptix module
py::str     module_filepath = py::module::import("PyOptix").attr("__file__");
py::object  Path            = py::module::import("pathlib").attr("Path");
std::string ray_file        = py::str(Path(module_filepath).attr("parent").attr("joinpath")("ptx_files", "ray_programs.ptx")).cast<std::string>();

// context and graph hierarchy
int                  deviceID;
optix::Context       ctx;
optix::Group         top_group;
optix::GeometryGroup gg;
optix::Material      scene_mat;

// input buffers
optix::Buffer origins_buffer;
optix::Buffer directions_buffer;

// output buffers
optix::Buffer d_buffer;
optix::Buffer uv_buffer;
optix::Buffer object_index_buffer;
optix::Buffer tri_index_buffer;

// output buffers for shadow rays
optix::Buffer shadow_buffer;

void createContext()
{
	if (!ctx) {
		CUDACHECKERROR(cudaGetDevice(&deviceID));

		const int RTX = true;
		if (rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(RTX), &RTX) != RT_SUCCESS) throw std::runtime_error("RTX mode not available");

		ctx = optix::Context::create();
		ctx->setRayTypeCount(1);
		ctx->setEntryPointCount(3);
		ctx->setStackSize(1);
		ctx["scene_epsilon"]->setFloat(1e-3f);

		// create the material and program objects
		optix::Program exception_program = ctx->createProgramFromPTXFile(ray_file, "exception");

		// the index is refering to the entry point
		ctx->setExceptionProgram(0, exception_program);
		ctx->setRayGenerationProgram(0, ctx->createProgramFromPTXFile(ray_file, "create_rays"));
		ctx->setRayGenerationProgram(1, ctx->createProgramFromPTXFile(ray_file, "create_shadow_rays"));
		ctx->setRayGenerationProgram(2, ctx->createProgramFromPTXFile(ray_file, "create_infinite_shadow_rays"));

		// the index is refering to ray type
		ctx->setMissProgram(0, ctx->createProgramFromPTXFile(ray_file, "miss"));
		scene_mat = ctx->createMaterial();
		scene_mat->setClosestHitProgram(0, ctx->createProgramFromPTXFile(ray_file, "closest_hit"));

		// create the graph hierarchy
		top_group = ctx->createGroup();
		top_group->setAcceleration(ctx->createAcceleration("NoAccel"));

		gg = ctx->createGeometryGroup();
		gg->setAcceleration(ctx->createAcceleration("Trbvh"));

		top_group->addChild(gg);
		ctx["top_object"]->set(top_group);
	}
}

void createBufferForTensor(optix::Buffer& fillBuf, torch::Tensor t)
{
	CHECK_INPUT(t);

	if (t.size(-1) > 4) throw std::invalid_argument("Tensor would require more than 4 floats per element!");

	// create object, if not initialized
	if (!fillBuf) fillBuf = ctx->createBufferForCUDA(RT_BUFFER_INPUT);

	// set the size accordingly if not already set correctly
	RTsize w;
	fillBuf->getSize(w);
	if (int64_t(w) != (t.numel() / t.size(-1))) fillBuf->setSize(t.numel() / t.size(-1));

	if (t.dtype() == torch::kFloat32) {
		if (t.dim() == 1 || t.size(-1) == 1)
			fillBuf->setFormat(RT_FORMAT_FLOAT);
		else {
			switch (t.size(-1)) {
			case 2: fillBuf->setFormat(RT_FORMAT_FLOAT2); break;
			case 3: fillBuf->setFormat(RT_FORMAT_FLOAT3); break;
			case 4: fillBuf->setFormat(RT_FORMAT_FLOAT4); break;
			}
		}
	} else if (t.dtype() == torch::kInt32) {
		// hack for index buffers: declare as UINT type
		if (t.dim() == 1 || t.size(-1) == 1)
			fillBuf->setFormat(RT_FORMAT_UNSIGNED_INT);
		else {
			switch (t.size(-1)) {
			case 2: fillBuf->setFormat(RT_FORMAT_UNSIGNED_INT2); break;
			case 3: fillBuf->setFormat(RT_FORMAT_UNSIGNED_INT3); break;
			case 4: fillBuf->setFormat(RT_FORMAT_UNSIGNED_INT4); break;
			}
		}
	} else
		throw std::invalid_argument("Tensor not of required dtype");

	// use the memory provided by the torch tensors as optix::Buffer objects
	fillBuf->setDevicePointer(deviceID, t.data_ptr());
}

void addMeshTriangleSoup(torch::Tensor vertex_buffer)
{
	if (!ctx) createContext();

	optix::Buffer vertexBuffer;
	createBufferForTensor(vertexBuffer, vertex_buffer);

	auto mesh_geometry = ctx->createGeometryTriangles();
	mesh_geometry->setFlagsPerMaterial(0, RT_GEOMETRY_FLAG_NONE);
	mesh_geometry->setBuildFlags(RT_GEOMETRY_BUILD_FLAG_NONE);

	mesh_geometry->setVertices(vertex_buffer.numel() / vertex_buffer.size(-1), vertexBuffer, RT_FORMAT_FLOAT3);
	mesh_geometry->setPrimitiveCount(vertex_buffer.numel() / (vertex_buffer.size(-1) * vertex_buffer.size(-2)));

	auto geom_inst = ctx->createGeometryInstance();
	geom_inst->addMaterial(scene_mat);
	geom_inst->setGeometryTriangles(mesh_geometry);
	geom_inst["object_index"]->setInt(gg->getChildCount());

	gg->addChild(geom_inst);
}

void addMeshIndexed(torch::Tensor vertex_buffer, torch::Tensor index_buffer)
{
	if (!ctx) createContext();

	optix::Buffer vertexBuffer;
	createBufferForTensor(vertexBuffer, vertex_buffer);

	optix::Buffer indexBuffer;
	createBufferForTensor(indexBuffer, index_buffer);

	auto mesh_geometry = ctx->createGeometryTriangles();
	mesh_geometry->setFlagsPerMaterial(0, RT_GEOMETRY_FLAG_NONE);
	mesh_geometry->setBuildFlags(RT_GEOMETRY_BUILD_FLAG_NONE);

	mesh_geometry->setVertices(vertex_buffer.numel() / vertex_buffer.size(-1), vertexBuffer, RT_FORMAT_FLOAT3);
	mesh_geometry->setTriangleIndices(indexBuffer, RT_FORMAT_UNSIGNED_INT3);
	mesh_geometry->setPrimitiveCount(index_buffer.numel());

	auto geom_inst = ctx->createGeometryInstance();
	geom_inst->addMaterial(scene_mat);
	geom_inst->setGeometryTriangles(mesh_geometry);
	geom_inst["object_index"]->setInt(gg->getChildCount());

	gg->addChild(geom_inst);
}

void resizeOutputBuffers(RTsize width)
{
	if (!d_buffer) d_buffer = ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, width);
	if (!uv_buffer) uv_buffer = ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT2, width);
	if (!object_index_buffer) object_index_buffer = ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_LONG_LONG, width);
	if (!tri_index_buffer) tri_index_buffer = ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_LONG_LONG, width);
	if (!shadow_buffer) shadow_buffer = ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_BYTE, width);

	RTsize w;
	d_buffer->getSize(w);
	if (w != width) d_buffer->setSize(width);

	uv_buffer->getSize(w);
	if (w != width) uv_buffer->setSize(width);

	object_index_buffer->getSize(w);
	if (w != width) object_index_buffer->setSize(width);

	tri_index_buffer->getSize(w);
	if (w != width) tri_index_buffer->setSize(width);

	shadow_buffer->getSize(w);
	if (w != width) shadow_buffer->setSize(width);
}

torch::Tensor queryPossibleHit(torch::Tensor origins, torch::Tensor directions, unsigned int objectIndex)
{
	if (origins.sizes() != directions.sizes() || origins.size(-1) != 3) throw std::invalid_argument("Ray Tensor sizes don't match");

	if (!ctx) createContext();

	createBufferForTensor(origins_buffer, origins);
	createBufferForTensor(directions_buffer, directions);

	// set arguments for program invocation
	ctx["ray_origins"]->set(origins_buffer);
	ctx["ray_directions"]->set(directions_buffer);

	resizeOutputBuffers(origins.numel() / origins.size(-1));
	ctx["shadow_buffer"]->set(shadow_buffer);

	// temporarily change the scene hierarchy with only the selected element being traced
	if (objectIndex > gg->getChildCount()) throw std::invalid_argument("Object index is not referring to a valid child");

	auto obj       = gg->getChild(objectIndex);
	auto tmp_group = ctx->createGeometryGroup();
	tmp_group->setAcceleration(ctx->createAcceleration("NoAccel"));
	tmp_group->addChild(obj);
	top_group->setChild(0, tmp_group);

	// TODO: check only, if validation is necessary
	ctx->validate();

	// start the trace
	ctx->launch(2, origins.numel() / origins.size(-1));
	auto options = origins.options().dtype(torch::kChar);

	auto          ret_sizes    = origins.sizes().slice(0, origins.sizes().size() - 1);
	torch::Tensor shadowTensor = torch::empty(ret_sizes, options);
	CUDACHECKERROR(cudaMemcpy(shadowTensor.data_ptr(), shadow_buffer->getDevicePointer(deviceID), shadowTensor.nbytes(), cudaMemcpyDeviceToDevice));

	// reset the scene hierarchy
	top_group->setChild(0, gg);

	return shadowTensor;
}

std::vector<torch::Tensor> traceRays(torch::Tensor origins, torch::Tensor directions, int rayType)
{
	if (origins.sizes() != directions.sizes() || origins.size(-1) != 3) throw std::invalid_argument("Ray Tensor sizes don't match");

	if (!ctx) createContext();

	createBufferForTensor(origins_buffer, origins);
	createBufferForTensor(directions_buffer, directions);

	// set arguments for program invocation
	ctx["ray_origins"]->set(origins_buffer);
	ctx["ray_directions"]->set(directions_buffer);

	resizeOutputBuffers(origins.numel() / origins.size(-1));
	ctx["d_buffer"]->set(d_buffer);
	ctx["uv_buffer"]->set(uv_buffer);
	ctx["object_index_buffer"]->set(object_index_buffer);
	ctx["tri_index_buffer"]->set(tri_index_buffer);
	ctx["shadow_buffer"]->set(shadow_buffer);

	// TODO: check only, if validation is necessary
	ctx->validate();

	// start the trace
	ctx->launch(rayType, origins.numel() / origins.size(-1));

	auto ret_sizes = origins.sizes().slice(0, origins.sizes().size() - 1);
	if (rayType == 0) {
		// define output tensors
		auto options = origins.options().dtype(torch::kFloat32);

		auto uv_ret_sizes = ret_sizes.vec();
		uv_ret_sizes.push_back(2);
		torch::Tensor depthTensor = torch::empty(ret_sizes, options);
		torch::Tensor uvTensor    = torch::empty(uv_ret_sizes, options);

		// to be able to index into other buffers, type has to be long
		auto          longOptions       = options.dtype(torch::kLong);
		torch::Tensor objectIndexTensor = torch::empty(ret_sizes, longOptions);
		torch::Tensor triIndexTensor    = torch::empty(ret_sizes, longOptions);

		// do memcpy
		CUDACHECKERROR(cudaMemcpy(depthTensor.data_ptr(), d_buffer->getDevicePointer(deviceID), depthTensor.nbytes(), cudaMemcpyDeviceToDevice));
		CUDACHECKERROR(cudaMemcpy(uvTensor.data_ptr(), uv_buffer->getDevicePointer(deviceID), uvTensor.nbytes(), cudaMemcpyDeviceToDevice));
		CUDACHECKERROR(cudaMemcpy(objectIndexTensor.data_ptr(), object_index_buffer->getDevicePointer(deviceID), objectIndexTensor.nbytes(), cudaMemcpyDeviceToDevice));
		CUDACHECKERROR(cudaMemcpy(triIndexTensor.data_ptr(), tri_index_buffer->getDevicePointer(deviceID), triIndexTensor.nbytes(), cudaMemcpyDeviceToDevice));

		return {depthTensor, uvTensor, objectIndexTensor, triIndexTensor};
	} else if (rayType == 1) {
		auto options = origins.options().dtype(torch::kChar);

		torch::Tensor shadowTensor = torch::empty(ret_sizes, options);
		CUDACHECKERROR(cudaMemcpy(shadowTensor.data_ptr(), shadow_buffer->getDevicePointer(deviceID), shadowTensor.nbytes(), cudaMemcpyDeviceToDevice));

		return {shadowTensor};
	} else {
		throw std::invalid_argument("Ray Type unknown");
	}
}

void updateSceneGeometry(torch::Tensor vertex_buffer, unsigned int childIdx)
{
	if (!ctx) throw std::invalid_argument("Can't update scene geometry without even having a context object!");

	// maybe this update part is unnecessary, but if the location of the vertices has been moved, it is necessary
	optix::Buffer vertexBuffer;
	createBufferForTensor(vertexBuffer, vertex_buffer);

	auto mesh_geometry = ctx->createGeometryTriangles();
	mesh_geometry->setFlagsPerMaterial(0, RT_GEOMETRY_FLAG_NONE);
	mesh_geometry->setBuildFlags(RT_GEOMETRY_BUILD_FLAG_NONE);
	mesh_geometry->setVertices(vertex_buffer.numel() / vertex_buffer.size(-1), vertexBuffer, RT_FORMAT_FLOAT3);
	mesh_geometry->setPrimitiveCount(vertex_buffer.numel() / (vertex_buffer.size(-1) * vertex_buffer.size(-2)));

	gg->getChild(childIdx)->setGeometryTriangles(mesh_geometry);

	// Mark acceleration structure as needing to be rebuilt
	gg->getAcceleration()->markDirty();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("trace_rays", &traceRays, "Trace Rays and return closest hit information");

	m.def("query_possible_hit", &queryPossibleHit, "Isolates an object and reports if rays would hit this object");

	m.def("add_mesh", &addMeshIndexed, "Adds a mesh to the scene with vertex and index buffer");

	m.def("add_mesh", &addMeshTriangleSoup, "Adds a mesh to the scene with only vertex buffer");

	m.def("update_scene_geometry", &updateSceneGeometry, "Marks the scene acceleration structure as dirty, forcing a rebuild on the next trace");

	m.def(
	    "get_module_file", []() { return module_filepath; }, "Get the current module file");

	m.def(
	    "get_ptx_files", []() { return ray_file; }, "Get the ptx files associated with this module");
}
