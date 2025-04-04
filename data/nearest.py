import open3d as o3d
def find_nearest_points(points, mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)  
    scene = o3d.t.geometry.RaycastingScene()  
    scene.add_triangles(mesh)  
    return scene.compute_closest_points(points)['points']  
