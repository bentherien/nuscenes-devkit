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

RESULTS_NAME='predicted_tracks_trainval'
CLASS_ID = 'car'
VERSION='v1.0-trainval'
SPLIT = 'val'
DURATION_AFTER_ERROR = 10

class ErrorVisualizer:
    def __init__(self, error, nusc=None, preds=None, error_tag=None, show_gt=True, 
                 show_pd=True, width=1280, height=960, zoom=0.125, point_color='intensity',
                 point_cmap='Greys', point_size=2.5, line_width=1, save_dir='vis'):
        
        if nusc is None:
            self.nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
        else:
            self.nusc = nusc

        if error_tag == 'neg_pos_pair':
            self.error_info = nusc.get('instance', error['negative']['prev_object_id'])
            self.objects_of_interest = [error['negative']['curr_object_id'], error['negative']['prev_object_id']]
            self.pred_of_interest = [error['negative']['hypothesis_id']]
            if error['hypothesis_switch']:
                self.pred_of_interest.append(error['negative2']['hypothesis_id'])
        if SPLIT == 'val':
            split = set(splits.val)
        elif SPLIT == 'train':
            split = set(splits.train)
        else:
            NotImplementedError
        self.scene = list(filter(lambda s: s['name'] in split, self.nusc.scene))
        self.scene_names_to_idx = {}
        for idx in range(len(self.scene)): 
            self.scene_names_to_idx[self.scene[idx]['token']] = idx
        
        if isinstance(preds, dict):
            self.preds = preds['results']
        elif isinstance(preds, str):
            with open(preds, 'r') as f:
                self.preds = json.load(f)['results']
        else:
            raise ValueError

        self.error = error
        self.error_tag = error_tag
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
        size = [dim/2 for dim in size]
        box.transform(np.diag([*size, 1]))
        box.rotate(rot)
        box.translate(translation)
        return box

    def get_gt_boxes(self, sample):
        boxes = []
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            # Only visualize objects of interest
            # if ann['instance_token'] not in self.objects_of_interest:
            #     continue
            # Only visualize car
            if CLASS_ID not in ann['category_name']:
                continue
            if ann['instance_token'] in self.objects_of_interest:
                self.lookat = ann['translation']
            box = self.convert_nus_bbox(ann['translation'], ann['rotation'], ann['size'])
            box.paint_uniform_color([1, 0, 0])
            boxes.append(box)
        return boxes

    def get_pd_boxes(self, sample_token):
        boxes = []
        # Specific coloring for cases
        color_list = [[0,0,1], [0.5,1,0.5]]
        color_id = 0
        for pred in self.preds[sample_token]:
            # Only visualize predictions of interest
            # if pred['tracking_id'] not in self.pred_of_interest:
            #     continue
            if CLASS_ID not in pred['tracking_name']:
                continue
            box = self.convert_nus_bbox(pred['translation'], pred['rotation'], pred['size'])

            # if pred['tracking_id'] not in self.id_color_map:
            #     self.id_color_map[pred['tracking_id']] = \
            #         colorsys.hsv_to_rgb(np.random.rand(), 1, 0.7)
            # box.paint_uniform_color(self.id_color_map[pred['tracking_id']])
            # text.paint_uniform_color(self.id_color_map[pred['tracking_id']])

            box.paint_uniform_color([0,0,0])
            if pred['tracking_id'] in self.pred_of_interest:
                text = self.get_text_3d(str(int(pred['tracking_id'])), pos=pred['translation'], rot=pred['rotation'])
                box.paint_uniform_color(color_list[color_id])
                text.paint_uniform_color(color_list[color_id])
                color_id += 1
                boxes.append(text)

            boxes.append(box)
            
        return boxes

    def get_text_3d(self, text, pos, rot, degree=0.0, font='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', font_size=100):
        """
        Generate a 3D text point cloud used for visualization.
        :param text: content of the text
        :param pos: 3D xyz position of the text upper left corner
        :param direction: 3D normalized direction of where the text faces
        :param degree: in plane rotation of text
        :param font: Name of the font - change it according to your system
        :param font_size: size of the font
        :return: o3d.geoemtry.PointCloud object
        """

        from PIL import Image, ImageFont, ImageDraw
        from pyquaternion import Quaternion

        font_obj = ImageFont.truetype(font, font_size)
        font_dim = font_obj.getsize(text)

        img = Image.new('RGB', font_dim, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
        img = np.asarray(img)
        img_mask = img[:, :, 0] < 128
        indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
        pcd.points = o3d.utility.Vector3dVector(indices / 100.0)
        pcd.translate(pos)
        return pcd

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
        vc.set_lookat(self.lookat)
    
        # Set the camera parameters to get a bird's eye view
        front = [0, 0, -1]  # Camera facing downwards
        lookat = [0, 1, 0]  # Camera looking at the origin from above
        up = [-0.5, 0.25, 0]      # Camera's up direction
        up /= np.linalg.norm(up)
        vc.set_up(up)
        vc.set_zoom(self.zoom)

        # Update render options
        ro = vis.get_render_option()
        vis.poll_events()
        vis.update_renderer()
        ro.point_size = self.point_size
    
        if self.save_dir is not None:
            filename = f'{self.save_dir}/{sample_token}.png'
            vis.capture_screen_image(filename, do_render=True)
        return False

    def __getitem__(self, scene_id):
        self.id_color_map = dict()
        if scene_id not in self.scene_names_to_idx.keys():
            return None
        scene_idx = self.scene_names_to_idx[scene_id]
        scene = self.scene[scene_idx]
        obj_sample_tokens = []
        sample_tokens = []
        sample_token = scene['first_sample_token']
        obj_pov = error['negative']['prev_object_id']

        # Find first and last frame where the object exists 
        first_frame_found = False
        while sample_token != '':
            for anno in self.nusc.get('sample', sample_token)['anns']:
                obj_id = nusc.get('sample_annotation', anno)['instance_token']
                if obj_pov == obj_id:
                    obj_sample_tokens.append(sample_token)
            sample_tokens.append(sample_token)
            sample_token = self.nusc.get('sample', sample_token)['next']
        if len(sample_tokens) == 0:
            raise Exception("Object of Interest Not Found in Scene")
        first_frame = obj_sample_tokens[0]
        last_frame_idx = min(self.error['frame_num'] + DURATION_AFTER_ERROR, scene['nbr_samples']-1)
        last_frame = sample_tokens[last_frame_idx] # five after?

        # Reset and get visualization
        sample_tokens = []
        sample_token = first_frame
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        while sample_token != last_frame and sample_token != '':
            sample_tokens.append(sample_token)
            self.render_sample(vis, sample_token)
            sample_token = self.nusc.get('sample', sample_token)['next']
        return sample_tokens

def render_video(sample_tokens, save_dir='vis', filename='untitled.mp4'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    images = [f'{sample_token}.png' for sample_token in sample_tokens]
    frame = cv2.imread(os.path.join(save_dir, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(os.path.join(save_dir, filename), cv2.VideoWriter_fourcc(*'mp4v'), 2, (width,height))
    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(save_dir, image)))
        os.remove(os.path.join(save_dir, image))
    cv2.destroyAllWindows()
    video.release()

def get_errors(error_path):

    # Load error of interest
    with open(error_path, "r") as error_file:
        error_json = json.load(error_file)
    thresholds = list(error_json[CLASS_ID].keys())
    thresholds.sort()
    return error_json, thresholds

if __name__ == '__main__':
    nusc = NuScenes(version=f'{VERSION}', dataroot='dataset/nuscenes', verbose=True)
    error_json, thresholds = get_errors(error_path=f'results/{RESULTS_NAME}/neg_pos_pair.json')
    # only care about lowest threshold
    thresholds = [thresholds[0]]
    for THRESH in thresholds:
        print(f'Saving visuals for: {THRESH}')
        save_dir = f'vis/{RESULTS_NAME}/vis-{VERSION}-{CLASS_ID}'
        for i in range(len(error_json[CLASS_ID][THRESH])):
            error = error_json[CLASS_ID][THRESH][i]
            vis = ErrorVisualizer(error, nusc=nusc, preds=f'tracks/{RESULTS_NAME}.json', error_tag='neg_pos_pair', save_dir=save_dir)
            scene_id = error['scene_id']
            sample_tokens = vis[scene_id]
            if sample_tokens:
                render_video(sample_tokens, filename=f'tracking_scene_{VERSION}_{i}.mp4', save_dir=save_dir)