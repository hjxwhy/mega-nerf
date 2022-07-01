import open3d as o3d
import json
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
import torch
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from mega_nerf.misc_utils import main_tqdm, main_print
np.random.seed(11)

def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],  # frustum origin
                               [-half_w, -half_h, frustum_length],  # top-left image corner
                               [half_w, -half_h, frustum_length],  # top-right image corner
                               [half_w, half_h, frustum_length],  # bottom-right image corner
                               [-half_w, half_h, frustum_length]])  # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i + 1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
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


def visualize_cameras(colored_camera_dicts, sphere_radius, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

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
            frustums.append(get_camera_frustum(img_size, K, W2C, frustum_length=camera_size, color=color))
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

def read_cameras(metadata_paths, pose_scale_factor=1, origin=[0, 0, 0]):
    camera_dicts = {}
    camera_positions = []
    for meta_path in metadata_paths:
        camera_data = torch.load(meta_path, map_location='cpu')
        intrinsic = camera_data['intrinsics'].numpy()
        H = camera_data['H']
        W = camera_data['W']
        K = [[intrinsic[0], 0, intrinsic[2], 0], [0, intrinsic[1], intrinsic[3], 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        c2w = camera_data['c2w']
        c2w[:3, 3] *= pose_scale_factor
        c2w[:3, 3] += origin
        camera_positions.append(c2w[:3, 3].numpy())
        w2c = torch.inverse(torch.cat([c2w, torch.tensor([[0,0,0,1]])])).numpy()
        camera_dicts[meta_path.stem] = {
            'K': K,
            'W2C': w2c.flatten(),
            'img_size': [W, H],
            'camera_size': 1
        }
    return camera_dicts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="Path to data.", default='/home/hjx/Documents/building-pixsfm')
    args = parser.parse_args()
    ray_altitude_range = [8, 50]

    dataset_path = Path(args.dataset_path)
    coordinate_info = torch.load(dataset_path / 'coordinates.pt', map_location='cpu')
    origin_drb = coordinate_info['origin_drb']
    pose_scale_factor = coordinate_info['pose_scale_factor']

    ray_altitude_range = [(x - origin_drb[0]) / pose_scale_factor for x in ray_altitude_range]
    main_print('Ray altitude range: {}->{}'.format(ray_altitude_range[0], ray_altitude_range[1]))

    metadata_paths = list((dataset_path / 'train' / 'metadata').iterdir()) \
                     + list((dataset_path / 'val' / 'metadata').iterdir())
    # print(metadata_paths)
    camera_dicts = read_cameras(metadata_paths, pose_scale_factor, origin_drb)
    colored_camera = [([0, 1, 0], camera_dicts)]
    visualize_cameras(colored_camera_dicts=colored_camera, sphere_radius=1)

    # camera_positions = torch.cat([torch.load(x, map_location='cpu')['c2w'][:3, 3].unsqueeze(0) for x in metadata_paths])
    # main_print('Number of images in dir: {}'.format(camera_positions.shape))

    # min_position = camera_positions.min(dim=0)[0]
    # max_position = camera_positions.max(dim=0)[0]
