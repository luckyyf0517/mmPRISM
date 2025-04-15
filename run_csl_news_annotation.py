import os
import cv2
import time
import torch
import argparse
import numpy as np
from termcolor import colored
from tqdm import tqdm
from glob import glob
from src.utils.plot import plot_hand_camera
from scipy.ndimage import gaussian_filter1d

from demo.hamer.vitpose_model import ViTPoseModel
from demo.hamer.hamer.configs import CACHE_DIR_HAMER
from demo.hamer.hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from demo.hamer.hamer.utils import recursive_to
from demo.hamer.hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from demo.hamer.hamer.utils.renderer import Renderer, cam_crop_to_full
from demo.hamer.hamer.utils.utils_detectron2 import DefaultPredictor_Lazy


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
camera_params = {
    'camera_position': [0, 0, 0],
    'camera_target': [0, 0, 1],
    'camera_intrinsic': [454.788/2.5, 454.696/2.5, 321.283/2.5, 186.695/2.5]
}


def process_single_image(img, model, model_cfg, detector, keypoint_detector, renderer, device='cuda'):
    """
    Process a single image and return hand mesh reconstruction results
    Args:
        img: numpy array of shape [H, W, 3] in BGR format
        model: HaMeR model
        model_cfg: model config
        detector: body detector (VitDet)
        keypoint_detector: keypoint detector (ViTPose)
        renderer: mesh renderer
        device: device to run model on
    Returns:
        dict containing reconstruction results, or None if no hands detected
    """
    # Detect humans in image
    img_bgr = img[:, :, ::-1]
    det_out = detector(img_bgr)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
    # Detect human keypoints
    vitposes_out = keypoint_detector.predict_pose(img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],)
    assert len(vitposes_out) == 1, 'Must be only one human in the image'
    
    left_hand_keyp = vitposes_out[0]['keypoints'][-42:-21]
    right_hand_keyp = vitposes_out[0]['keypoints'][-21:]
    
    def process_hand(keyp):
        valid = keyp[:,2] > 0.5
        assert sum(valid) > 3, 'No valid hand keypoints'
        bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
        return bbox
    
    boxes = np.stack([process_hand(left_hand_keyp), process_hand(right_hand_keyp)])
    right = np.stack([0, 1])

    # # Draw two bounding boxes on the image
    # img_bgr_copy = img_bgr.copy()
    # for bbox in boxes:
    #     x1, y1, x2, y2 = map(int, bbox)
    #     cv2.rectangle(img_bgr_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2
    # # Save the image with bounding boxes
    # cv2.imwrite('output.png', img_bgr_copy)

    # Extract hand bounding boxes
    batch_size = 2
    dataset = ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))
    batch = recursive_to(batch, device)
    
    with torch.no_grad():
        out = model(batch)
        
    multiplier = (2*batch['right']-1)
    pred_cam = out['pred_cam']
    pred_cam[:,1] = multiplier*pred_cam[:,1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

    verts = out['pred_vertices'].detach().cpu().numpy()  # [B, V, 3]
    joints = out['pred_keypoints_3d'].detach().cpu().numpy()  # [B, J, 3]
    is_right = batch['right'].cpu().numpy()  # [B]
    verts[..., 0] = (2 * is_right[:, None] - 1) * verts[..., 0]
    verts = verts + pred_cam_t_full[:, None, :]
    joints[..., 0] = (2 * is_right[:, None] - 1) * joints[..., 0]
    joints = joints + pred_cam_t_full[:, None, :]
    hand_pose = out['pred_mano_params']['hand_pose'].detach().cpu().numpy()
    betas = out['pred_mano_params']['betas'].detach().cpu().numpy()
    return {'verts': verts, 'joints': joints, 'hand_pose': hand_pose, 'betas': betas}
    

def process_sequence(seq_id, model, model_cfg, detector, keypoint_detector, renderer, device='cuda'):
    data_root = os.path.join('/root/autodl-tmp/datasets/CollectedProc', seq_id)
    if not os.path.exists(os.path.join(data_root, 'color.npy')):
        print(colored(f'{seq_id} does not exist', 'red'))
        return
    
    color_data_list = np.load(os.path.join(data_root, 'color.npy'))
    depth_data_list = np.load(os.path.join(data_root, 'depth.npy')) # used for correcting the z-axis
    num_frames = len(color_data_list)
    
    # stage 1: get pred results, and save the depth
    results_list = []
    depth = np.zeros((num_frames, 2))
    for index in tqdm(range(num_frames)):
        color_img = color_data_list[index]
        depth_img = depth_data_list[index]
        results = process_single_image(color_img, model, model_cfg, detector, keypoint_detector, renderer, device='cuda')
        for k in range(2):
            x_pred = results['joints'][k][:, 0]
            y_pred = results['joints'][k][:, 1]
            z_pred = results['joints'][k][:, 2]
            x2d = (x_pred / z_pred * 12500 + 321.283).astype(np.int32)
            y2d = (y_pred / z_pred * 12500 + 186.695).astype(np.int32)
            joints2d = np.stack([x2d, y2d], axis=-1)
            joints2d = joints2d[joints2d[:, 0] > 0]
            joints2d = joints2d[joints2d[:, 0] < 640]
            joints2d = joints2d[joints2d[:, 1] > 0]
            joints2d = joints2d[joints2d[:, 1] < 360]
            depths = depth_img[joints2d[:, 1], joints2d[:, 0]]
            depth[index, k] = depths[(100 < depths) & (depths < 500)].mean() / 1e3
        results_list.append(results)
    
    # stage 1.5: smooth the depth
    depth_smooth = gaussian_filter1d(depth, sigma=1, axis=0)
    
    # stage 2: correct the z-axis
    for index in tqdm(range(num_frames)):
        color_img = color_data_list[index]
        results = results_list[index]
        joints_all = np.zeros((2, 21, 3))
        verts_all = np.zeros((2, 778, 3))
        # process both hands
        for k in range(2):
            joints = results['joints'][k].copy()
            depth_pred = joints[:, 2].mean()
            diff = depth_smooth[index, k] - depth_pred

            # def compute_rigid_transform(source, target):
            #     """rotate first, then translate"""
            #     source_centered = source - source[0]
            #     target_centered = target - target[0]
            #     H = np.dot(source_centered.T, target_centered)
            #     U, S, Vt = np.linalg.svd(H)
            #     R = np.dot(Vt.T, U.T)
            #     det = np.linalg.det(R)
            #     if det < 0:
            #         Vt[-1] *= -1
            #         R = np.dot(Vt.T, U.T)
            #     translation = target.mean(0) - source.mean(0)
            #     return R, translation
            
            # # correct the joints
            # joints_org = results['joints'][k].copy()
            # joints_target = results['joints'][k].copy()
            # scale = (joints_org[:, 2] + diff) / joints_org[:, 2]
            
            # joints_target[:, 0] *= scale * 12500 / 454.788
            # joints_target[:, 1] *= scale * 12500 / 454.696
            # joints_target[:, 2] *= scale
            # R, translation = compute_rigid_transform(joints_org, joints_target)
            # rot_center = joints_org.mean(0)
            
            # joints = results['joints'][k].copy()
            # joints = np.einsum('ij,kj->ki', R, joints_org - rot_center) + rot_center + translation
            # joints_all[k] = joints
            
            # # correct the verts (use the same rotation and translation)
            # verts = results['verts'][k].copy()
            # verts = np.einsum('ij,kj->ki', R, verts - rot_center) + rot_center + translation
            # verts_all[k] = verts
            
            # correct the joints
            joints_target = results['joints'][k].copy()
            scale = (joints_target[:, 2] + diff) / joints_target[:, 2]
            joints_target[:, 0] *= scale * 12500 / 454.788
            joints_target[:, 1] *= scale * 12500 / 454.696
            joints_target[:, 2] *= scale
            joints_all[k] = joints_target
            
            verts_target = results['verts'][k].copy()
            scale = (verts_target[:, 2] + diff) / verts_target[:, 2]
            verts_target[:, 0] *= scale * 12500 / 454.788
            verts_target[:, 1] *= scale * 12500 / 454.696
            verts_target[:, 2] *= scale
            verts_all[k] = verts_target
        
        # plot the hand
        image = cv2.resize(color_img, (256, 144))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plot_hand_camera(image, joints_all[0], **camera_params, boundary=False)
        plot_hand_camera(image, joints_all[1], **camera_params, boundary=False)
        os.makedirs(f'/root/autodl-tmp/mmHand/output/{seq_id}', exist_ok=True)
        cv2.imwrite(f'/root/autodl-tmp/mmHand/output/{seq_id}/{index:06d}.png', image)
        
        # save the joints and verts
        os.makedirs(f'{data_root}/joints', exist_ok=True)
        os.makedirs(f'{data_root}/verts', exist_ok=True)
        np.save(f'{data_root}/joints/{index:06d}.npy', joints_all)
        np.save(f'{data_root}/verts/{index:06d}.npy', verts_all) 
        # save the params
        hand_pose = results['hand_pose']
        betas = results['betas']
        params = {
            'hand_pose': hand_pose,
            'betas': betas}
        os.makedirs(f'{data_root}/params', exist_ok=True)
        np.savez(f'{data_root}/params/{index:06d}.npz', **params)


if __name__ == '__main__':
    # load the hamer model
    os.chdir('demo/hamer')
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--id', nargs='+', default=None, help='List of sequences to process')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    if args.id is None:
        args.id = ['%04d' % i for i in range(args.start, args.end + 1)]

    model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    model = model.to('cuda')
    model.eval()
    # Load detector
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        cfg_path = 'hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    # keypoint detector
    keypoint_detector = ViTPoseModel('cuda')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    print(colored('All models loaded', 'green'))
    
    # start annotation 
    failed_seq_list = []
    
    for seq_id in tqdm(args.id):
        print(colored(f'Processing {seq_id}', 'yellow'))
        try: 
            process_sequence(seq_id, model, model_cfg, detector, keypoint_detector, renderer, device='cuda')
        except Exception as e:
            failed_seq_list.append(seq_id)
            print(colored(f'Failed to process {seq_id}', 'red'))
            print(e)

    if len(failed_seq_list) > 0:
        print(colored(f'Failed seq list: {failed_seq_list}', 'red'))
    else: 
        print(colored('All sequences processed successfully', 'green'))
