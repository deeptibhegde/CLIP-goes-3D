import open3d as o3d


def load_mesh(filepath):
    return o3d.io.read_triangle_mesh(filepath)


def export_mesh(mesh, filepath):
    o3d.io.write_triangle_mesh(filepath, mesh)


def load_pcd(filepath):
    return o3d.io.read_point_cloud(filepath)


def export_pcd(pcd, filepath):
    o3d.io.write_point_cloud(filepath, pcd)


def mesh_to_pcd(mesh, number_of_points=2048):
    return mesh.sample_points_uniformly(number_of_points=number_of_points)
