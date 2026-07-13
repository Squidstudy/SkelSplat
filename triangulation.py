import os
import numpy as np
from tqdm import tqdm
from arguments.config_handler import TriangulationConfigHandler
from scene.dataset_readers import DataLoader
import hydra
from omegaconf import DictConfig
import sys
import logging
from scipy.spatial.transform import Rotation as Rotation
import json
import open3d as o3d


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

############ H36M ############
def get_calibration_matrices_h36m(camera_data):

    camera_names = ["54138969", "55011271", "58860488", "60457274"]
    K = []
    for cam in camera_names:
        calibration_matrix = camera_data["intrinsics"].get(cam, {}).get("calibration_matrix", None)
        K.append(np.array(calibration_matrix).reshape(3, 3))
    return K


def get_extrinsics_h36m(camera_data, subject_id):

    camera_names = ["54138969", "55011271", "58860488", "60457274"]
    R = []
    t = []  # 回傳的是「原始的 R/t 數值」（沒有經過四元數轉換的慣例調整）
    # P = K[R|t]: P = K[R|t]
    
    for cam in camera_names:
        extrinsics = camera_data["extrinsics"].get(subject_id, {}).get(cam, {})
        rotation = extrinsics.get("R", None)
        translation = extrinsics.get("t", None)
        R.append(np.array(rotation).reshape(3, 3)) # 從一維陣列轉成3*3 <see note>
        t.append(np.array(translation).reshape(3, 1))

    return R, t

def create_projection_matrix_h36m(K_list, R_list, t_list):

    P = []

    for (k, r, t) in zip(K_list, R_list, t_list):
        RT = np.hstack((r, t.reshape(-1, 1)))  # Combine rotation and translation
        P.append(np.dot(k, RT))  # Projection matrix = K * [R | t]
        # 把 R 跟 t 橫向拼接成 3×4 矩陣，再跟 3×3 的 K 相乘得到 3×4 的投影矩
    return P

############ OCCLUSION-PERSON ############
def get_camera_parameters_op(camera_data, nviews):
    
    camera_ids = ["0", "1", "2", "3", "4", "5", "6", "7"]
    cameras = camera_ids[1::2][:nviews]
    K = {}
    R = {}
    t = {}

    for cam in cameras:
        cam = int(cam)
        camera = camera_data[cam]
        fx, fy = camera["fx"], camera["fy"]
        cx, cy = camera["cx"], camera["cy"]
        K[cam] = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
        R[cam] = np.array(camera["R"]).reshape(3, 3)
        t[cam] = np.array(camera["T"]).reshape(3, 1)
        t[cam] = -R[cam] @ t[cam] # utils/graphics_utils.py 取出裡面的相機中心座標轉「平移向量」的慣例轉換
        # （因為原始資料存的 T 定義方式跟三角測量公式需要的定義不一樣）
    
    return K, R, t 


############ PANOPTIC ############
def get_camera_parameters_panoptic(camera_data, nviews):
    
    camera_names = ["00_01", "00_02", "00_10", "00_13", "00_03", "00_23", "00_19", "00_30"][:nviews]
    K = {}
    R = {}
    t = {}

    for cam in camera_names:
        for data in camera_data["cameras"]:
            if data["name"] == cam:
                K[cam] = np.array(data["K"]).reshape(3, 3)
                R[cam] = np.array(data["R"]).reshape(3, 3)
                t[cam] = np.array(data["t"]).reshape(3, 1) * 10 # from cm to mm => 跟其他資料集的座標單位統一
    
    return K, R, t

#  panoptic/occlusion-person
def create_projection_matrix(K_dict, R_dict, t_dict):
    P = []
    for cam in sorted(K_dict.keys()): # 這兩個資料集的相機參數是存成字典而不是固定順序的清單
        K = K_dict[cam]
        R = R_dict[cam]
        t = t_dict[cam]
        RT = np.hstack((R, t.reshape(-1, 1)))  # Combine rotation and translation
        P.append(np.dot(K, RT))  # Projection matrix = K * [R | t]
    return P

# DLT + SVD 演算法
def triangulate_points_multi_camera(P_list, x_list):
    # P_list:每台相機各自的投影矩陣(形狀 3×4,把 3D 點映射到該相機 2D 畫面的矩陣,前面 create_projection_matrix 算出來的東西)
    # x_list:每台相機各自觀測到的 2D 像素座標 (x, y),順序要跟 P_list 一一對應(同一個索引代表同一台相機)。
    
    A = []

    for P, x in zip(P_list, x_list):
        # 把每台相機的 (x, y) 轉成齊次座標 (x, y, 1)——多補一個 1,方便底下用矩陣運算表示「相差一個未知縮放係數」的關係。
        x_hom = np.append(x, 1)  # Convert (x, y) to homogeneous (x, y, 1)
        # Direct Linear Transform: 2D->3D
        A.append(x_hom[0] * P[2, :] - P[0, :])
        A.append(x_hom[1] * P[2, :] - P[1, :])

    A = np.array(A)  

    # Solve using SVD, 要解的是 A @ X = 0 這種齊次線性方程組。因為現實資料有雜訊,不太可能存在一個 X 讓 AX 完全等於 0,所以退而求其次:找一個 ||X||=1 的 X,讓 ||AX|| 盡量小(最小平方意義下的最佳解)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]  # Last row is solution
    X = X / X[3]  # Vt[-1] 算出來的 X 是齊次座標 (X, Y, Z, W),對應到真實 3D 歐式座標是 (X/W, Y/W, Z/W)

    return X


def triangulate_poses(P_list, poses_2d):
    num_joints = poses_2d.shape[1] #  因為 poses_2d 是 GPU tensor，三角測量用的是 numpy／SVD 運算，需要先搬回 CPU 並脫離計算圖）
    X_3D = []

    for j in range(num_joints):
        x_list = [poses_2d[v, j, :2].detach().cpu().numpy() for v in range(len(P_list))]
        X_3D.append(triangulate_points_multi_camera(P_list, x_list))

    return np.array(X_3D)


def triangulation(dataset, dataset_loader, output_dir, log):

    if "h36m" in dataset.data_root:
        metadata_path = os.path.join("data/h36m/3d_gt", "cameras", "camera-parameters.json")
        with open(metadata_path, "r") as file:
            camera_data = json.load(file) 

        K_cameras = get_calibration_matrices_h36m(camera_data)

    if "occlusion-person" in dataset.data_root:
        metadata_path = os.path.join(dataset.data_root, "cameras.json")
        with open(metadata_path, "r") as file:
            camera_data = json.load(file)

    # Same as DataLoader
    log.info(f"{len(dataset_loader)} scenes to process")
    camera_data_pan = {}

    for scene_id, scene_data in dataset_loader:

        pose_3d, pose_3d_gt, poses_2d, cameras, scene_name = scene_data
        log.info(f"Processing scene {scene_name}")
        subject_id = scene_name.split("_")[0]

        if "h36m" in dataset.data_root:
            R_cameras, t_cameras = get_extrinsics_h36m(camera_data, subject_id)
            P = create_projection_matrix_h36m(K_cameras, R_cameras, t_cameras)
        if "occlusion-person" in dataset.data_root:
            K_cameras, R_cameras, t_cameras = get_camera_parameters_op(camera_data[str(scene_id)], dataset.nviews)
            P = create_projection_matrix(K_cameras, R_cameras, t_cameras)
        if "panoptic" in dataset.data_root:
            activity = scene_name.split("_")[1] + "_" + scene_name.split("_")[2]
            camera_data_path = os.path.join(dataset.data_root, "3d_gt", "cameras", f"calibration_{activity}.json")
            with open(camera_data_path, "r") as file:
                camera_data_pan[activity] = json.load(file)

            K_cameras, R_cameras, t_cameras = get_camera_parameters_panoptic(camera_data_pan[activity], dataset.nviews)
            P = create_projection_matrix(K_cameras, R_cameras, t_cameras)

        pose_3d_triang = triangulate_poses(P, poses_2d[:, :, :2])

        pose_3d_norm = pose_3d_triang[:, :3] / pose_3d_triang[:, 3].reshape(-1, 1)

        pcd = o3d.geometry.PointCloud()  # 建立點雲物件、寫成 .ply 檔，存進 point_cloud/iteration_0 資料夾
        pcd.points = o3d.utility.Vector3dVector(pose_3d_norm)

        point_cloud_path = os.path.join(output_dir, "point_cloud/iteration_0")
        os.makedirs(point_cloud_path, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(point_cloud_path, f"{scene_name}.ply"), pcd)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    config = TriangulationConfigHandler(cfg)

    output_dir = config.hydra_out
    dataset = cfg.dataset
    debug = cfg.debug

    print(output_dir)

    log = logging.getLogger(__name__)

    initial_guess_path = os.path.join(dataset.data_root, "initial_guess", dataset.initial_guess)
    poses_2d_path = os.path.join(dataset.data_root, "2d_" + dataset.poses_2d)

    dataset_loader = DataLoader(dataset.data_root, initial_guess_path, poses_2d_path,
                                frame_step=dataset.frame_step, start_id=dataset.start_scene_id,
                                end_id=dataset.end_scene_id, nviews=dataset.nviews)

    triangulation(dataset, dataset_loader, output_dir, log)

if __name__ == "__main__":
    main()
