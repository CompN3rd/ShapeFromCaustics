import torch as th
import torch.nn.functional as F
import numpy as np
import PyOptix as popt
import pywavefront as pw
import typing as tp
from collections import defaultdict


from .utils import gram_schmidt, barycentric_interpolate, normalize_tensor, dot_product, barycentric_slerp

params = {"Light_Source": {"roughness": 1, "emissivity": [1, 1, 1]},
          "Target": {"roughness": 0.01, "specularity": [1, 0.921, 0.725]},
          "Stand": {"roughness": 1},
          "Light": {"roughness": 1}
          }


def read_scene(filepath, light_name="Light_Source", idx_offset=0, gpu_scene=None, mat_buffer_factory=None):
    scene = pw.Wavefront(filepath, collect_faces=True)
    if any(x.vertex_format != "T2F_N3F_V3F" for x in scene.materials.values()):
        raise ValueError("Expected vertex format is T2F_N3F_V3F")

    if gpu_scene is None:
        gpu_scene = PyOptixScene()
    for idx, (mesh_name, mesh) in enumerate(scene.meshes.items()):
        vertex_buffer = np.stack([mesh.materials[0].vertices[5 + i::8] for i in range(3)], axis=-1).astype(np.float32).reshape(-1, 3, 3)
        normal_buffer = np.stack([mesh.materials[0].vertices[2 + i::8] for i in range(3)], axis=-1).astype(np.float32).reshape(-1, 3, 3)
        texcoord_buffer = np.stack([mesh.materials[0].vertices[i::8] for i in range(2)], axis=-1).astype(np.float32).reshape(-1, 3, 2)

        if mat_buffer_factory is None:
            m = PyOptixObject(mesh_name, idx + idx_offset,
                              *map(lambda x: th.from_numpy(x).cuda(), [vertex_buffer, normal_buffer, texcoord_buffer]),
                              material_buffers={x: th.from_numpy(np.asarray(y, dtype=np.float32)).view(1, -1, 1, 1).cuda() for x, y in params[mesh_name].items()}
                              )
        else:
            m = PyOptixObject(mesh_name, idx + idx_offset,
                              *map(lambda x: th.from_numpy(x).cuda(), [vertex_buffer, normal_buffer, texcoord_buffer]),
                              material_buffers=mat_buffer_factory()
                              )

    gpu_scene.add_object(m)
    if light_name is not None and mesh_name == light_name:
        light_index = idx

    if light_name is not None:
        return gpu_scene, light_index
    else:
        return gpu_scene


def grid_shuffle(v, p, flip=False):
    v_names, p_names = v.names, p.names
    v.rename_(None), p.rename_(None)
    if v.dim() == 2:
        # lower left triangles
        v[::2, 0] = p[:-1, :-1].reshape(-1)
        v[::2, 1] = p[:-1, 1:].reshape(-1)
        v[::2, 2] = p[1:, :-1].reshape(-1)

        # upper right triangles
        v[1::2, 0] = p[:-1, 1:].reshape(-1)
        v[1::2, 1] = p[1:, 1:].reshape(-1)
        v[1::2, 2] = p[1:, :-1].reshape(-1)
    else:
        num_dim = v.size(-1)
        # lower left triangles
        v[::2, 0] = p[:-1, :-1].reshape(-1, num_dim)
        v[::2, 1 if not flip else 2] = p[:-1, 1:].reshape(-1, num_dim)
        v[::2, 2 if not flip else 1] = p[1:, :-1].reshape(-1, num_dim)

        # upper right triangles
        v[1::2, 0] = p[:-1, 1:].reshape(-1, num_dim)
        v[1::2, 1 if not flip else 2] = p[1:, 1:].reshape(-1, num_dim)
        v[1::2, 2 if not flip else 1] = p[1:, :-1].reshape(-1, num_dim)

    v.rename_(*v_names)
    p.rename_(*p_names)


def create_from_height_field(coords: th.Tensor, height_field: th.Tensor, normal_field: th.Tensor, sensor_height: float, additional_elements=[]):
    gpu_scene = PyOptixScene()

    if coords is not None and height_field is not None and normal_field is not None:
        points = th.cat((coords.align_to(..., 'dim'), height_field.align_to(..., 'dim')), dim='dim').rename(None)

        # height field mesh
        num_plane_triangles = 2 * (coords.size('width') - 1) * (coords.size('height') - 1)
        num_front_back_triangles = 2 * (coords.size('width') - 1)
        num_left_right_triangles = 2 * (coords.size('height') - 1)
        substrate_vertices = th.zeros(2 * num_plane_triangles + 2 * num_front_back_triangles + 2 * num_left_right_triangles, 3, 3, dtype=coords.dtype, device=coords.device)  # .refine_names('triangle','vertex','dim')
        substrate_normals = th.zeros(2 * num_plane_triangles + 2 * num_front_back_triangles + 2 * num_left_right_triangles, 3, 3, dtype=coords.dtype, device=coords.device)  # .refine_names('triangle','vertex','dim')
        substrate_texcoords = th.zeros(2 * num_plane_triangles + 2 * num_front_back_triangles + 2 * num_left_right_triangles, 3, 2, dtype=coords.dtype, device=coords.device)  # .refine_names('triangle','vertex','dim')

        # create substrate
        # top plane
        start_offset = 0
        end_offset = num_plane_triangles
        grid_shuffle(substrate_vertices[start_offset:end_offset], points)
        grid_shuffle(substrate_normals[start_offset:end_offset], normal_field.align_to(..., 'dim').rename(None))
        grid_shuffle(substrate_texcoords[start_offset:end_offset], 0.5 * points[:, :, :2] - 0.5)

        # bottom plane
        start_offset = end_offset
        end_offset = start_offset + num_plane_triangles
        bottom_points = F.pad(coords.align_to(..., 'dim').rename(None), (0, 1))  # pad with zeros in z
        grid_shuffle(substrate_vertices[start_offset: end_offset], bottom_points, flip=True)
        grid_shuffle(substrate_normals[start_offset: end_offset], th.tensor([[[0, 0, -1]]], dtype=normal_field.dtype, device=normal_field.device).expand_as(points), flip=True)
        grid_shuffle(substrate_texcoords[start_offset: end_offset], 0.5 * points[:, :, :2] - 0.5, flip=True)

        # side planes
        # front
        start_offset = end_offset
        end_offset = start_offset + num_front_back_triangles
        grid_shuffle(substrate_vertices[start_offset: end_offset], th.cat((bottom_points[0:1], points[0:1]), dim=0))
        grid_shuffle(substrate_normals[start_offset: end_offset], th.tensor([[[0, -1, 0]]], dtype=normal_field.dtype, device=normal_field.device).expand(2, coords.size('width'), -1))
        grid_shuffle(substrate_texcoords[start_offset: end_offset], 0.5 * th.cat((points[0:1, :, :2], points[0:1, :, :2]), dim=0) - 0.5)

        # back
        start_offset = end_offset
        end_offset = start_offset + num_front_back_triangles
        grid_shuffle(substrate_vertices[start_offset: end_offset], th.cat((bottom_points[-1:], points[-1:]), dim=0), flip=True)
        grid_shuffle(substrate_normals[start_offset: end_offset], th.tensor([[[0, 1, 0]]], dtype=normal_field.dtype, device=normal_field.device).expand(2, coords.size('width'), -1), flip=True)
        grid_shuffle(substrate_texcoords[start_offset: end_offset], 0.5 * th.cat((points[-1:, :, :2], points[-1:, :, :2]), dim=0) - 0.5, flip=True)

        # left
        start_offset = end_offset
        end_offset = start_offset + num_left_right_triangles
        grid_shuffle(substrate_vertices[start_offset: end_offset], th.cat((bottom_points[:, 0:1], points[:, 0:1]), dim=1))
        grid_shuffle(substrate_normals[start_offset: end_offset], th.tensor([[[-1, 0, 0]]], dtype=normal_field.dtype, device=normal_field.device).expand(2, coords.size('height'), -1))
        grid_shuffle(substrate_texcoords[start_offset: end_offset], 0.5 * th.cat((points[:, 0:1, :2], points[:, 0:1, :2]), dim=1) - 0.5)

        # right
        start_offset = end_offset
        end_offset = start_offset + num_left_right_triangles
        grid_shuffle(substrate_vertices[start_offset: end_offset], th.cat((bottom_points[:, -1:], points[:, -1:]), dim=1), flip=True)
        grid_shuffle(substrate_normals[start_offset: end_offset], th.tensor([[[1, 0, 0]]], dtype=normal_field.dtype, device=normal_field.device).expand(2, coords.size('height'), -1), flip=True)
        grid_shuffle(substrate_texcoords[start_offset: end_offset], 0.5 * th.cat((points[:, -1:, :2], points[:, -1:, :2]), dim=1) - 0.5, flip=True)

        gpu_scene.add_object(PyOptixObject('height_field_mesh', 0,
                                           substrate_vertices, substrate_normals, substrate_texcoords,
                                           material_buffers={'roughness': th.tensor([[[[1e-7]]]], dtype=coords.dtype, device=coords.device)}))
    else:
        if additional_elements is not None:
            for elem in additional_elements:
                read_scene(elem, light_name=None, idx_offset=len(gpu_scene), gpu_scene=gpu_scene, mat_buffer_factory=lambda: {'roughness': th.tensor([[[[1e-7]]]]).cuda()})

    # ground plane mesh
    if coords is not None:
        min_bounds = coords.flatten(['height', 'width'], 'coord').min(dim='coord')[0].rename(None)
        max_bounds = coords.flatten(['height', 'width'], 'coord').max(dim='coord')[0].rename(None)
    else:
        min_bounds = th.tensor([-1, -1], dtype=th.float32).cuda()
        max_bounds = th.tensor([1, 1], dtype=th.float32).cuda()

    ground_vertices = th.full((2, 3, 3), sensor_height, dtype=min_bounds.dtype, device=min_bounds.device)

    ground_vertices[0, 0, :2] = min_bounds
    ground_vertices[0, 1, :2] = th.as_tensor([max_bounds[0], min_bounds[1]], dtype=min_bounds.dtype, device=min_bounds.device)
    ground_vertices[0, 2, :2] = th.as_tensor([min_bounds[0], max_bounds[1]], dtype=min_bounds.dtype, device=min_bounds.device)

    ground_vertices[1, 0, :2] = th.as_tensor([min_bounds[0], max_bounds[1]], dtype=min_bounds.dtype, device=min_bounds.device)
    ground_vertices[1, 1, :2] = th.as_tensor([max_bounds[0], min_bounds[1]], dtype=min_bounds.dtype, device=min_bounds.device)
    ground_vertices[1, 2, :2] = max_bounds

    ground_normals = th.tensor([0, 0, 1], dtype=ground_vertices.dtype, device=ground_vertices.device).view(1, 1, 3).expand_as(ground_vertices)
    ground_texcoords = 0.5 * ground_vertices[:, :, :2] - 0.5

    gpu_scene.add_object(PyOptixObject('ground_mesh', len(gpu_scene),
                                       ground_vertices, ground_normals, ground_texcoords,
                                       material_buffers={'photon_map': th.zeros(1, 1, 1, 1, dtype=ground_vertices.dtype, device=ground_vertices.device)}))

    # add additional elements based on list parameter
    if coords is not None and additional_elements is not None:
        for elem in additional_elements:
            read_scene(elem, light_name=None, idx_offset=len(gpu_scene), gpu_scene=gpu_scene, mat_buffer_factory=lambda: {'roughness': th.tensor([[[[1e-7]]]], dtype=ground_vertices.dtype, device=ground_vertices.device)})

    return gpu_scene


class PyOptixObject:
    def __compute_tangent_bitangent(self):
        # compute tangent and bitangent
        # http://www.terathon.com/code/tangent.html
        # rhs.shape = T x v x c x 2
        rhs = th.zeros(self._normals.shape + (2,), dtype=self._normals.dtype, device=self._normals.device)
        for i in range(self._normals.shape[1]):
            rhs[:, i, :, 0] = self._vertices[:, (i + 1) % 3] - self._vertices[:, i]
            rhs[:, i, :, 1] = self._vertices[:, (i + 2) % 3] - self._vertices[:, i]

        # st_mat.shape = T x v x 2 x 2
        st_mat = th.zeros(self._normals.shape[:2] + (2, 2), dtype=self._normals.dtype, device=self._normals.device)
        for i in range(self._normals.shape[1]):
            st_mat[:, i, 0, 0] = self._texcoords[:, (i + 2) % 3, 1] - self._texcoords[:, i, 1]
            st_mat[:, i, 1, 1] = self._texcoords[:, (i + 1) % 3, 0] - self._texcoords[:, i, 0]
            st_mat[:, i, 0, 1] = -self._texcoords[:, (i + 2) % 3, 0] + self._texcoords[:, i, 0]
            st_mat[:, i, 1, 0] = -self._texcoords[:, (i + 1) % 3, 1] + self._texcoords[:, i, 1]
        determinant = st_mat[:, :, 1, 1] * st_mat[:, :, 0, 0] - st_mat[:, :, 0, 1] * st_mat[:, :, 1, 0]
        res_mat = th.matmul(rhs, st_mat) / determinant.unsqueeze(-1).unsqueeze(-1)

        # res_mat.shape = T x v x c x 2
        self._tangents = res_mat[:, :, :, 0]
        self._bitangents = res_mat[:, :, :, 1]

        # Gram-Schmidt orthonormalization
        ret = gram_schmidt(self._normals, self._tangents, self._bitangents, dim=-1)
        self._normals = ret[0]
        self._tangents = ret[1]
        self._bitangents = ret[2]

    def __compute_L_planes(self):
        geom_normal = self.geometric_normal()

        orgs = self._vertices[:, [1, 2, 0]]
        L = th.cross(geom_normal.unsqueeze(1).expand_as(self._vertices), self._vertices[:, [2, 0, 1]] - orgs, dim=-1)

        # compute distance from origin
        d = dot_product(orgs, orgs, dim=-1, keepdim=True)

        # norm such that vertex across has distance 1
        # shouldn't result in nan's if triangle is non-degenerate
        factor = dot_product(L, self._vertices, dim=-1, keepdim=True) + d

        self._L = th.cat((L, d), dim=-1) / factor

    def __init__(self, name, scene_idx, vertex_buffer, normal_buffer, texcoord_buffer, material_buffers):
        self._name = name
        self._index = scene_idx
        # necessary buffers
        self._vertices = vertex_buffer
        self._normals = normal_buffer
        self._texcoords = texcoord_buffer

        # Igehy1999 for normal interpolated triangles 'barycentric planes'
        self._L = None

        # also share data with Optix Framework
        popt.add_mesh(self._vertices)

        # material buffers for generic materials
        for i, (buffer_name, value) in enumerate(material_buffers.items()):
            # set the default value
            if i == 0:
                self.material_buffers = defaultdict(lambda: th.tensor([0], dtype=value.dtype, device=value.device))
            self.material_buffers[buffer_name] = value

        self.__compute_tangent_bitangent()

    def update_from_height_field(self, coords: th.Tensor, height_field: th.Tensor, height_field_normals: th.Tensor):
        points = th.cat((coords.align_to(..., 'dim'), height_field.align_to(..., 'dim')), dim='dim').rename(None)
        num_plane_triangles = 2 * (coords.size('width') - 1) * (coords.size('height') - 1)
        num_front_back_triangles = 2 * (coords.size('width') - 1)
        num_left_right_triangles = 2 * (coords.size('height') - 1)
        self._vertices = th.zeros(2 * num_plane_triangles + 2 * num_front_back_triangles + 2 * num_left_right_triangles, 3, 3, dtype=coords.dtype, device=coords.device)  # .refine_names('triangle','vertex','dim')
        self._normals = th.zeros(2 * num_plane_triangles + 2 * num_front_back_triangles + 2 * num_left_right_triangles, 3, 3, dtype=coords.dtype, device=coords.device)  # .refine_names('triangle','vertex','dim')
        self._texcoords = th.zeros(2 * num_plane_triangles + 2 * num_front_back_triangles + 2 * num_left_right_triangles, 3, 2, dtype=coords.dtype, device=coords.device)  # .refine_names('triangle','vertex','dim')

        # create substrate
        # top plane
        start_offset = 0
        end_offset = num_plane_triangles
        grid_shuffle(self._vertices[start_offset: end_offset], points)
        grid_shuffle(self._normals[start_offset: end_offset], height_field_normals.align_to(..., 'dim').rename(None))
        grid_shuffle(self._texcoords[start_offset: end_offset], 0.5 * points[:, :, :2] - 0.5)

        # bottom plane
        start_offset = end_offset
        end_offset = start_offset + num_plane_triangles
        bottom_points = F.pad(coords.align_to(..., 'dim').rename(None), (0, 1))  # pad with zeros in z
        grid_shuffle(self._vertices[start_offset: end_offset], bottom_points, flip=True)
        grid_shuffle(self._normals[start_offset: end_offset], th.tensor([[[0, 0, -1]]], dtype=height_field_normals.dtype, device=height_field_normals.device).expand_as(points), flip=True)
        grid_shuffle(self._texcoords[start_offset: end_offset], 0.5 * points[:, :, :2] - 0.5, flip=True)

        # side planes
        # front
        start_offset = end_offset
        end_offset = start_offset + num_front_back_triangles
        grid_shuffle(self._vertices[start_offset: end_offset], th.cat((bottom_points[0:1], points[0:1]), dim=0))
        grid_shuffle(self._normals[start_offset: end_offset], th.tensor([[[0, -1, 0]]], dtype=height_field_normals.dtype, device=height_field_normals.device).expand(2, coords.size('width'), -1))
        grid_shuffle(self._texcoords[start_offset: end_offset], 0.5 * th.cat((points[0:1, :, :2], points[0:1, :, :2]), dim=0) - 0.5)

        # back
        start_offset = end_offset
        end_offset = start_offset + num_front_back_triangles
        grid_shuffle(self._vertices[start_offset: end_offset], th.cat((bottom_points[-1:], points[-1:]), dim=0), flip=True)
        grid_shuffle(self._normals[start_offset: end_offset], th.tensor([[[0, 1, 0]]], dtype=height_field_normals.dtype, device=height_field_normals.device).expand(2, coords.size('width'), -1), flip=True)
        grid_shuffle(self._texcoords[start_offset: end_offset], 0.5 * th.cat((points[-1:, :, :2], points[-1:, :, :2]), dim=0) - 0.5, flip=True)

        # left
        start_offset = end_offset
        end_offset = start_offset + num_left_right_triangles
        grid_shuffle(self._vertices[start_offset: end_offset], th.cat((bottom_points[:, 0:1], points[:, 0:1]), dim=1))
        grid_shuffle(self._normals[start_offset: end_offset], th.tensor([[[-1, 0, 0]]], dtype=height_field_normals.dtype, device=height_field_normals.device).expand(2, coords.size('height'), -1))
        grid_shuffle(self._texcoords[start_offset: end_offset], 0.5 * th.cat((points[:, 0:1, :2], points[:, 0:1, :2]), dim=1) - 0.5)

        # right
        start_offset = end_offset
        end_offset = start_offset + num_left_right_triangles
        grid_shuffle(self._vertices[start_offset: end_offset], th.cat((bottom_points[:, -1:], points[:, -1:]), dim=1), flip=True)
        grid_shuffle(self._normals[start_offset: end_offset], th.tensor([[[1, 0, 0]]], dtype=height_field_normals.dtype, device=height_field_normals.device).expand(2, coords.size('height'), -1), flip=True)
        grid_shuffle(self._texcoords[start_offset: end_offset], 0.5 * th.cat((points[:, -1:, :2], points[:, -1:, :2]), dim=1) - 0.5, flip=True)

        self.__compute_tangent_bitangent()

        popt.update_scene_geometry(self._vertices, self._index)

        # Igehy1999 for normal interpolated triangles 'barycentric planes'
        self._L = None

    def sample_random(self, position_tensor: th.Tensor, sample_directions=False):
        # position_tensor.shape = N x c
        # triangle index sampling should be proportional to subtended angle
        # angles.shape = T x N
        angles = F.relu(self._subtended_angle(position_tensor, True), inplace=True)
        angles_sum = angles.sum(0)
        mask = angles_sum.gt(0)
        # draw samples that are proportional to the angles
        triangle_index = th.zeros(position_tensor.size(0), dtype=th.int64, device=position_tensor.device)
        triangle_index[mask] = th.multinomial(angles[:, mask].permute(1, 0), 1)[:, 0]

        # TODO: random_barys not really uniform wrt. subtended angle of triangle
        random_barys = th.rand(position_tensor.size(0), 2, dtype=position_tensor.dtype, device=position_tensor.device)
        su0 = th.sqrt(random_barys[:, 0])
        random_barys[:, 0] = 1 - su0
        random_barys[:, 1] = random_barys[:, 1] * su0

        # avoid division by zero (this assumes that the sample at this point returns zero, because of backface culling)
        pdf = th.ones_like(angles_sum)
        pdf[mask] = (1 / angles_sum[mask])
        if not sample_directions:
            return triangle_index, random_barys, pdf
        else:
            # project the drawn triangle to unit sphere
            # self._vertices[triangle_index].shape = N x 3 x c
            normalized_corners = normalize_tensor(self._vertices[triangle_index] - position_tensor.unsqueeze(1))
            dir_tensor = barycentric_slerp(normalized_corners, random_barys)

            return dir_tensor, pdf

    def subtended_angle(self, position_tensor: th.Tensor, return_uncumulated=False):
        """
        Oosterom-Strackee-Formula: https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
        returns the subtended angle of the whole mesh as seen from points in position_tensor
        """
        # position_tensor.shape = N x c
        # self._vertices.shape = T x v x c
        # diffs.shape = T x v x N x c
        diffs = self._vertices.unsqueeze(-2) - position_tensor.unsqueeze(0).unsqueeze(0)
        # diff_norm.shape = T x v x N
        diffs_norm = th.norm(diffs, p=2, dim=-1)

        # get T x N values
        # we change the order in the cross product, to get positive values for cw triangles,
        # which are exported this way for our scene
        # numerators.shape = T x N
        numerators = dot_product(diffs[:, 0], th.cross(diffs[:, 2], diffs[:, 1], dim=-1))
        denominators = diffs_norm.prod(dim=1) + dot_product(diffs[:, 0], diffs[:, 1]) * diffs_norm[:, 2] + dot_product(diffs[:, 0], diffs[:, 2]) * diffs_norm[:, 1] + dot_product(diffs[:, 1], diffs[:, 2]) * diffs_norm[:, 0]

        # avoid undefined behavior (and NaNs in bwd)
        ret = th.zeros_like(numerators)
        mask = (th.logical_not(numerators.eq(0) & denominators.eq(0)))
        ret[mask] = 2 * th.atan2(numerators[mask], denominators[mask])

        # return one element for each element in position_tensor,
        # optionally reduce over the triangle dimension and ignore negative, i.e. backfacing triangles
        # size: T x N or N
        return ret if return_uncumulated else F.relu(ret, inplace=True).sum(0)

    def light_pdf(self, position_tensor, dir_tensor, evaluated_light_pdf=None):
        # position_tensor.shape = N x c
        # dir_tensor.shape = N x c
        # this assumes that dir_tensor is normalized
        # TODO: implement consistent backface culling: subtended_angle has it, query_possible_hit doesn't
        if evaluated_light_pdf is None:
            angles = self.subtended_angle(position_tensor, False)  # angles.shape = N
            angles_mask = angles.gt(0)
            evaluated_light_pdf = th.ones_like(angles)
            evaluated_light_pdf[angles_mask] = 1 / angles[angles_mask]

        possible_visibilies = popt.query_possible_hit(position_tensor, dir_tensor, self._index)
        return evaluated_light_pdf.view_as(possible_visibilies) * possible_visibilies.to(evaluated_light_pdf)

    def geometric_normal(self, cw=False):
        # self._vertices.shape = T x v x c
        if cw:
            normal = th.cross(self._vertices[:, 2] - self._vertices[:, 0], self._vertices[:, 1] - self._vertices[:, 0], dim=-1)
        else:
            normal = th.cross(self._vertices[:, 1] - self._vertices[:, 0], self._vertices[:, 2] - self._vertices[:, 0], dim=-1)
        return normalize_tensor(normal, dim=-1)

    def differential_normal(self, tri_index: th.Tensor, shading_normal: th.Tensor, *point_differentials: tp.Sequence[th.Tensor]):
        # shading_normal.shape = S x d
        # point_differentials.shape = S x d
        if self._L is None:
            self.__compute_L_planes()

        L_sample = self._L[tri_index, :, :shading_normal.size(-1)]
        dn_dx_list = (dot_product(dot_product(L_sample, pd.unsqueeze(1), dim=-1, keepdim=True), shading_normal.unsqueeze(1), dim=1) for pd in point_differentials)

        ndn = dot_product(shading_normal, shading_normal, dim=-1, keepdim=True)
        return ((ndn * dn_dx - dot_product(shading_normal, dn_dx, dim=-1, keepdim=True) * shading_normal) / (ndn**1.5) for dn_dx in dn_dx_list)


class PyOptixScene:
    def __init__(self):
        self._objects = th.jit.annotate(tp.List[PyOptixObject], [])
        self._dtype = None
        self._device = None

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, key: tp.Union[int, str]):
        if type(key) is int:
            return self._objects[key]
        elif type(key) is str:
            for mesh in self._objects:
                if mesh._name == key:
                    return mesh
            raise KeyError("Mesh with key {} not found".format(key))
        else:
            raise TypeError("Key type {} not supported".format(type(key)))

    def add_object(self, obj: PyOptixObject):
        self._objects.append(obj)
        if self._dtype is None or self._device is None:
            dummy_ref = next(iter(obj.material_buffers.values()))
            self._dtype = dummy_ref.dtype
            self._device = dummy_ref.device

    def get_light_mesh(self):
        # this assumes, that there is exactly one mesh with this property
        for mesh in self._objects:
            if mesh.material_buffers["emissivity"].gt(0).any().item():
                return mesh

    def differential_normal(self, object_index: th.Tensor, tri_index: th.Tensor, shading_normal: th.Tensor, *point_differentials: tp.Sequence[th.Tensor]):
        ret_vals = [th.zeros_like(pd) for pd in point_differentials]

        for ind, mesh in enumerate(self._objects):
            mesh_mask = object_index.eq(ind)
            if mesh_mask.any():
                ret_masks = mesh.differential_normal(tri_index[mesh_mask], shading_normal[mesh_mask], *(pd[mesh_mask] for pd in point_differentials))
                for r, rets in zip(ret_vals, ret_masks):
                    r[mesh_mask] = rets

        return ret_vals

    def prepare_hit_information(self, object_index, tri_index, uv, requested_params=None):
        ret_dict = defaultdict(lambda: th.tensor([0], dtype=self.dtype, device=self.device))

        # normal of triangle planes
        if requested_params is None or "geometric_normal" in requested_params:
            ret_dict["geometric_normal"] = th.zeros(uv.size(0), 3, dtype=uv.dtype, device=self._device)

        # special values that are always interpolated
        if requested_params is None or "normal" in requested_params:
            ret_dict["normal"] = th.zeros(uv.size(0), 3, dtype=uv.dtype, device=self._device)

        if requested_params is None or "tangent" in requested_params:
            ret_dict["tangent"] = th.zeros(uv.size(0), 3, dtype=uv.dtype, device=self._device)

        if requested_params is None or "bitangent" in requested_params:
            ret_dict["bitangent"] = th.zeros(uv.size(0), 3, dtype=uv.dtype, device=self._device)

        # collect all data from different objects into one big buffer
        for ind, mesh in enumerate(self._objects):
            mesh_mask = object_index.eq(ind)
            if mesh_mask.any():
                # look up geometric normal
                if requested_params is None or "geometric_normal" in requested_params:
                    ret_dict["geometric_normal"][mesh_mask] = mesh.geometric_normal()[tri_index[mesh_mask]]

                # interpolate all geometric information
                if requested_params is None or "normal" in requested_params:
                    ret_dict["normal"][mesh_mask] = barycentric_interpolate(mesh._normals, tri_index[mesh_mask], uv[mesh_mask])

                if requested_params is None or "tangent" in requested_params:
                    ret_dict["tangent"][mesh_mask] = barycentric_interpolate(mesh._tangents, tri_index[mesh_mask], uv[mesh_mask])

                if requested_params is None or "bitangent" in requested_params:
                    ret_dict["bitangent"][mesh_mask] = barycentric_interpolate(mesh._bitangents, tri_index[mesh_mask], uv[mesh_mask])

                # normalize texcoords to [-1; 1]
                texcoords_at_hit = (2 * barycentric_interpolate(mesh._texcoords, tri_index[mesh_mask], uv[mesh_mask]) - 1).unsqueeze(0).unsqueeze(0)

                # gather all lookup values
                for key, val in mesh.material_buffers.items():
                    if requested_params is None or key in requested_params:
                        # initialize
                        if key not in ret_dict:
                            ret_dict[key] = th.zeros(*(uv.size(0), val.size(1)) if val.size(1) > 1 else (uv.size(0),), dtype=val.dtype, device=val.device)
                        # query
                        ret_dict[key][mesh_mask] = F.grid_sample(val, texcoords_at_hit.to(val), align_corners=True).squeeze(0).squeeze(1).transpose(0, 1).squeeze(1)

        # geometric normals are already normalized, so no need to renormalize

        # renormalize other normals
        if requested_params is None or "normal" in requested_params:
            ret_dict["normal"] = normalize_tensor(ret_dict["normal"])

        if requested_params is None or "tangent" in requested_params:
            ret_dict["tangent"] = normalize_tensor(ret_dict["tangent"])

        if requested_params is None or "bitangent" in requested_params:
            ret_dict["bitangent"] = normalize_tensor(ret_dict["bitangent"])

        # cut negative values
        for key, val in ret_dict.items():
            # clamp roughness to epsilon value
            if key == "roughness":
                ret_dict[key].clamp_(min=1e-6)
                continue

            # if it is not geometric information
            if key not in ["normal", "tangent", "bitangent", "geometric_normal"]:
                ret_dict[key] = F.relu(val, inplace=True)

        return ret_dict
