import open3d as o3d
import numpy as np
import argparse
from pathlib import Path
import torch
import scipy
np.random.seed(11)


def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],  # frustum origin
                               # top-left image corner
                               [-half_w, -half_h, frustum_length],
                               # top-right image corner
                               [half_w, -half_h, frustum_length],
                               # bottom-right image corner
                               [half_w, half_h, frustum_length],
                               [-half_w, half_h, frustum_length]])  # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(
        1, 5)] + [[i, (i + 1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape(
        (1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(
        np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N * 5, 3))  # 5 vertices per frustum
    merged_lines = np.zeros((N * 8, 2))  # 8 lines per frustum
    merged_colors = np.zeros((N * 8, 3))  # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i * 5:(i + 1) * 5, :] = frustum_points
        merged_lines[i * 8:(i + 1) * 8, :] = frustum_lines + i * 5
        merged_colors[i * 8:(i + 1) * 8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset


def visualize_cameras(colored_camera_dicts, sphere_radius, geometry_file=None, geometry_type='mesh', centroid=None, lineset=None):
    # sphere = o3d.geometry.TriangleMesh.create_sphere(
    #     radius=sphere_radius, resolution=10)
    # sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    # sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=sphere_radius, origin=[0., 0., 0.])
    things_to_draw = [coord_frame]
    if lineset is not None:
        things_to_draw.append(lineset)
    if centroid is not None:
        for c in centroid:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=sphere_radius, origin=c)
            things_to_draw.append(coord_frame)

    idx = 0
    for color, camera_dict in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []
        for img_name in sorted(camera_dict.keys()):
            K = np.array(camera_dict[img_name]['K']).reshape((4, 4))
            W2C = np.array(camera_dict[img_name]['W2C']).reshape((4, 4))
            img_size = camera_dict[img_name]['img_size']
            camera_size = camera_dict[img_name]['camera_size']
            frustums.append(get_camera_frustum(
                img_size, K, W2C, frustum_length=camera_size, color=color))
            cnt += 1
        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


RDF_TO_DRB = torch.FloatTensor([[0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1]])


def read_cameras(metadata_paths, pose_scale_factor=1, origin=[0, 0, 0]):
    camera_dicts = {}
    camera_positions = []
    for meta_path in metadata_paths:
        camera_data = torch.load(meta_path, map_location='cpu')
        intrinsic = camera_data['intrinsics'].numpy()
        H = camera_data['H']
        W = camera_data['W']
        K = [[intrinsic[0], 0, intrinsic[2], 0], [
            0, intrinsic[1], intrinsic[3], 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        c2w = camera_data['c2w']

        #####################
        # 这个可视化是变回colmap最原始的pose的可视化，在这个数据集上，z轴不是正面朝下了，导致了变成y正面朝下，最后在变成x朝下，就很混乱
        # 如果不想看这个就直接注释掉就好了
        c2w = c2w[:, [1, 0, 2, 3]]
        c2w[:, 0] *= -1
        c2w[:3, 3] *= pose_scale_factor
        c2w[:3, 3] += origin

        c2w[:3, :3] = torch.inverse(RDF_TO_DRB) @ c2w[:3, :3] @ RDF_TO_DRB
        c2w[:3, 3:] = torch.inverse(RDF_TO_DRB) @ c2w[:3, 3:]
        #####################

        camera_positions.append(c2w[:3, 3].numpy())
        w2c = torch.inverse(
            torch.cat([c2w, torch.tensor([[0, 0, 0, 1]])])).numpy()
        camera_dicts[meta_path.stem] = {
            'K': K,
            'W2C': w2c.flatten(),
            'img_size': [W, H],
            'camera_size': 1
        }
    return camera_dicts


def scale_back(camera_poses, origin, scale_factor):
    camera_poses = camera_poses[:, [1, 0, 2, 3], ...]
    camera_poses[:, :1, ...] *= -1
    camera_poses[:, :3, 3] *= scale_factor
    camera_poses[:, :3, 3] += origin
    camera_poses[:, :3, :3] = camera_poses[:, :3, :3] @ RDF_TO_DRB.unsqueeze(0)
    # camera_poses[:, :3, 3:] = torch.inverse(RDF_TO_DRB).unsqueeze(0) @ camera_poses[:, :3, 3:]
    return camera_poses


def create_grid(p0, p1, p2, p3, ni1, ni2, color=(0, 0, 0)):
    '''
    p0, p1, p2, p3 : points defining a quadrilateral
    ni1: nb of equidistant intervals on segments p0p1 and p3p2
    ni2: nb of equidistant intervals on segments p1p2 and p0p3
    '''
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    vertices = [p0, p1, p2, p3]
    lines = [[0, 1], [0, 3], [1, 2], [2, 3]]
    for i in range(1, ni1):
        l = len(vertices)
        vertices.append((p0*(ni1-i)+p1*i)/ni1)
        vertices.append((p3*(ni1-i)+p2*i)/ni1)
        lines.append([l, l+1])
    for i in range(1, ni2):
        l = len(vertices)
        vertices.append((p1*(ni2-i)+p2*i)/ni2)
        vertices.append((p0*(ni2-i)+p3*i)/ni2)
        lines.append([l, l+1])
    vertices = o3d.utility.Vector3dVector(vertices)
    lines = o3d.utility.Vector2iVector(lines)
    mesh = o3d.geometry.LineSet(vertices, lines)
    return mesh


def vis(poses, centroids, grid_dim=[4, 2], camera_size=0.1):
    all_position = poses[:, :3, 3]
    min_position = all_position.min(0)
    max_position = all_position.max(0)
    distances = scipy.spatial.distance.cdist(
        all_position, centroids, 'minkowski', p=2)
    print(distances.shape)
    min_distance = distances.min(1)
    min_dist_ratio = distances / min_distance[:, None]
    ray_in_region = min_dist_ratio[:, 4] < 1.15

    region_pose = poses[ray_in_region, ...]
    other_pose = poses[np.logical_not(ray_in_region), ...]
    H = 1088
    W = 1632
    K = [[1253.5263072, 0, 792.2739649056, 0], [0, 1508.8347236479999,
                                                533.74918778879999, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    n = 0
    camera_dicts = {}
    for p in region_pose:
        camera_dicts[str(n)] = {
            'K': K,
            'W2C': np.linalg.inv(p),
            'img_size': [W, H],
            'camera_size': camera_size
        }
        n += 1
    camera_dicts2 = {}
    for p in other_pose:
        camera_dicts2[str(n)] = {
            'K': K,
            'W2C': np.linalg.inv(p),
            'img_size': [W, H],
            'camera_size': camera_size
        }
        n += 1

    lineset = create_grid(min_position, [max_position[0], min_position[1], 0], max_position, [
                          min_position[0], max_position[1], 0], grid_dim[0], grid_dim[1])
    colored_camera = [([0, 1, 0], camera_dicts), ([1, 0, 1], camera_dicts2)]

    visualize_cameras(colored_camera_dicts=colored_camera,
                      sphere_radius=1, centroid=centroids, lineset=lineset)


def read_dso_pose(dso_result):
    with open(dso_result, 'r') as f:
        lines = f.readlines()
        camera_poses = []
        for line in lines:
            line = line.strip().split()
            c2w = np.asarray(line[1:], dtype=np.float32).reshape(
                [4, 4])[None]
            camera_poses.append(c2w)
        camera_poses = np.vstack(camera_poses)
    return camera_poses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="Path to data.",
                        default='/home/hjx/Documents/mega_dji')
    parser.add_argument('--grid_dim', nargs='+', default=[2, 4])
    parser.add_argument('--vis_orig', default=True)
    args = parser.parse_args()
    # ray_altitude_range = [8, 50]
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_file():
        coordinate_info = torch.load(
            dataset_path / 'coordinates.pt', map_location='cpu')
        origin_drb = coordinate_info['origin_drb']
        pose_scale_factor = coordinate_info['pose_scale_factor']
        ray_altitude_range = coordinate_info['near_far']
        ray_altitude_range = [(x - origin_drb[2]) /
                              pose_scale_factor for x in ray_altitude_range]
        print(
            'Ray altitude range: {}->{}'.format(ray_altitude_range[0], ray_altitude_range[1]))

        metadata_paths = list((dataset_path / 'train' / 'metadata').iterdir()) \
            + list((dataset_path / 'val' / 'metadata').iterdir())
        camera_poses = torch.cat([torch.cat([torch.load(x, map_location='cpu')[
            'c2w'], torch.tensor([[0, 0, 0, 1]])]).unsqueeze(0) for x in metadata_paths])
    else:
        camera_poses = read_dso_pose(dataset_path)
        camera_poses = torch.from_numpy(camera_poses)
        camera_positions = camera_poses[:, :3, 3]
        min_position = camera_positions.min(0)[0]
        max_position = camera_positions.max(0)[0]
        orig = (min_position + max_position) * 0.5
        camera_poses[:, :3, 3] -= orig

    if args.vis_orig and not dataset_path.is_file():
        print('1111111')
        camera_poses = scale_back(camera_poses, origin_drb, pose_scale_factor)
        camera_positions = camera_poses[:, :3, 3]
        min_position = camera_positions.min(0)[0]
        max_position = camera_positions.max(0)[0]
        orig = (min_position + max_position) * 0.5
        camera_poses[:, :3, 3] -= orig
        args.grid_dim = [args.grid_dim[1], args.grid_dim[0]]

    camera_positions = camera_poses[:, :3, 3]

    min_position = camera_positions.min(0)[0]
    max_position = camera_positions.max(0)[0]
    print('Coord range: {} {}'.format(min_position, max_position))

    ranges = max_position[:-1] - min_position[:-1]
    offsets = [torch.arange(s) * ranges[i] / s + ranges[i] / (s * 2)
               for i, s in enumerate(args.grid_dim)]
    centroids = torch.stack((torch.ones((args.grid_dim[0], args.grid_dim[1])) * min_position[0],
                             torch.ones(
                                 (args.grid_dim[0], args.grid_dim[1])) * min_position[1],
                             torch.zeros((args.grid_dim[0], args.grid_dim[1])))).permute(1, 2, 0)
    centroids[:, :, 0] += offsets[0][:, None, ...]
    centroids[:, :, 1] += offsets[1]
    centroids = centroids.reshape(-1, 3)
    camera_size = 0.1
    vis(camera_poses.numpy(), centroids.numpy(), args.grid_dim, camera_size)
