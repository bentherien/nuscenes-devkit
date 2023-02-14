import os
import json
import colorsys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d
from nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm.autonotebook import tqdm
from scipy.spatial.transform import Rotation as R

class Visualizer:
    def __init__(self, nusc=None, preds='predicted_tracks_json.json', show_gt=True, show_pd=True,
                 width=640, height=480, zoom=0.2, point_color='intensity', point_cmap='Greys',
                 point_size=2.5, line_width=1, save_dir='vis'):
        if nusc is None:
            self.nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
        else:
            self.nusc = nusc
        val = set(splits.train)
        self.scene = list(filter(lambda s: s['name'] in val, self.nusc.scene))
        
        if isinstance(preds, dict):
            self.preds = preds['results']
        elif isinstance(preds, str):
            with open(preds, 'r') as f:
                self.preds = json.load(f)['results']
        else:
            raise ValueError

        self.show_gt = show_gt
        self.show_pd = show_pd
        self.width = width
        self.height = height
        self.zoom = zoom
        self.point_color = point_color
        self.point_cmap = point_cmap
        self.point_size = point_size
        self.line_width = line_width
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

    def get_pcd(self, sample_data):
        points = np.fromfile(f'{self.nusc.dataroot}/{sample_data["filename"]}', dtype=np.float32).reshape(-1, 5)
        sensor2ego = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
        ego2global = self.nusc.get('ego_pose', sample_data['ego_pose_token'])

        # Transform point cloud to global frame
        points[:,:3] = R.from_quat(np.roll(sensor2ego['rotation'], -1)).apply(points[:,:3])
        points[:,:3] += sensor2ego['translation']
        points[:,:3] = R.from_quat(np.roll(ego2global['rotation'], -1)).apply(points[:,:3])
        points[:,:3] += ego2global['translation']
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:,:3]))

        # Set point cloud colors
        if isinstance(self.point_color, (list, tuple, np.ndarray)):
            # Uniform RGB color
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(self.point_color, (points.shape[0], 1))[:,:3])
        elif self.point_color == 'intensity':
            pcd.colors = o3d.utility.Vector3dVector(
                cm.get_cmap(self.point_cmap)(np.log(points[:,3]+1)/np.log(256))[:,:3])
        elif self.point_color == 'height':
            pcd.colors = o3d.utility.Vector3dVector(
                cm.get_cmap(self.point_cmap)((points[:,2]+3)/10)[:,:3])
        else:
            raise ValueError
        return pcd

    def convert_nus_bbox(self, translation, rotation, size):
        # Use OrientedBoundingBox, cannot control line width
        rot = R.from_quat(np.roll(rotation, -1)).as_euler('zyx')[0]+np.pi/2
        rot = R.from_euler('z', rot).as_matrix()
        vertices = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
            [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ])
        edges = np.array([
            [0, 1], [0, 2], [1, 4], [2, 4],
            [3, 5], [3, 6], [5, 7], [6, 7],
            [0, 3], [1, 5], [2, 6], [4, 7]
        ], dtype=int)
        box = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector2iVector(edges))
        box.transform(np.diag([*size, 1]))
        box.rotate(rot)
        box.translate(translation)
        return box

    def get_gt_boxes(self, sample):
        boxes = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            # Only visualize car
            if ann['category_name'] != 'vehicle.car':
                continue
            box = self.convert_nus_bbox(ann['translation'], ann['rotation'], ann['size'])
            box.paint_uniform_color([1, 0, 0])
            boxes.append(box)
        return boxes

    def get_pd_boxes(self, sample_token):
        boxes = []
        for pred in self.preds[sample_token]:
            if pred['tracking_name'] != 'car':
                continue
            box = self.convert_nus_bbox(pred['translation'], pred['rotation'], pred['size'])
            if pred['tracking_id'] not in self.id_color_map:
                self.id_color_map[pred['tracking_id']] = \
                    colorsys.hsv_to_rgb(np.random.rand(), 1, 0.7)
            box.paint_uniform_color(self.id_color_map[pred['tracking_id']])
            boxes.append(box)
        return boxes

    def render_sample(self, vis, sample_token):
        sample = self.nusc.get('sample', sample_token)
        sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego2global = self.nusc.get('ego_pose', sample_data['ego_pose_token'])

        vis.clear_geometries()
        vis.add_geometry(self.get_pcd(sample_data))
        for box in self.get_pd_boxes(sample_token):
            if self.show_pd:
                vis.add_geometry(box)
        for box in self.get_gt_boxes(sample):
            if self.show_gt:
                vis.add_geometry(box)

        # Update view control
        vc = vis.get_view_control()
        lookat = ego2global['translation']
        vc.set_lookat(lookat)
        front = [-1, -1, 1]
        front /= np.linalg.norm(front)
        vc.set_front(front)
        up = np.cross(front, np.cross([0, 0, 1], front))
        up /= np.linalg.norm(up)
        vc.set_up(up)
        vc.set_zoom(self.zoom)

        # Update render options
        ro = vis.get_render_option()
        ro.point_size = self.point_size
    
        if self.save_dir is not None:
            filename = f'{self.save_dir}/{sample_token}.png'
            vis.capture_screen_image(filename, do_render=True)
        return False

    def __getitem__(self, idx):
        self.id_color_map = dict()
        scene = self.scene[idx]
        sample_tokens = []
        sample_token = scene['first_sample_token']
        def cb(vis):
            nonlocal sample_token
            if sample_token != '':
                sample_tokens.append(sample_token)
                self.render_sample(vis, sample_token)
                sample_token = self.nusc.get('sample', sample_token)['next']
            return False
        o3d.visualization.draw_geometries_with_animation_callback([], cb, width=self.width, height=self.height)
        return sample_tokens

def render_video(sample_tokens, save_dir='vis', filename='untitled.mp4'):
    images = [f'{sample_token}.png' for sample_token in sample_tokens]
    frame = cv2.imread(os.path.join(save_dir, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 2, (width,height))
    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(save_dir, image)))
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-mini', dataroot='dataset/nuscenes', verbose=False)
    vis = Visualizer(nusc=nusc, show_gt=False, preds='tracks/predicted_tracks_mini.json')
    scene_idx = 0
    sample_tokens = vis[scene_idx]
    render_video(sample_tokens, filename=f'tracking_scene{scene_idx}.mp4')