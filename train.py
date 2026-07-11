#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from random import randint
from gaussian_renderer import render_functions
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from arguments.config_handler import ConfigHandler
import PIL.Image as Image
from utils.general_utils import generate_heatmaps
from scene.dataset_readers import DataLoader
import hydra
from omegaconf import DictConfig
import sys
import logging
from utils import losses, early_stopping_strategy, consistency_losses
from utils.general_utils import unpack_covariance, OptEarlyStopping
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F

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

from itertools import combinations


def training(dataset, model, opt, pipe, debug, training, dataset_loader, output_dir, log):
    # 防止沒裝套對應套件
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # 字典[key]查詢語法 => 取代if/else
    opt_criterion = losses[training.loss_function] #查詢.yaml檔裡設定的loss_function是甚麼
    consistency_criterion = consistency_losses[training.consistency_loss]
    render = render_functions[pipe.rendering]
    early_stopping = early_stopping_strategy[training.early_stopping]() # ():把查到的類別立刻實例化成一個物件

    # 因為 optimization.random_background = False => 這一段不會被用到
    tb_writer = prepare_output_and_logger(output_dir)
    
    bg_color = [1, 1, 1] if model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # 對 DataLoader 索引出來的每一個scene各自跑一輪
    log.info(f"Training on {len(dataset_loader)} scenes")

    for scene_id, scene_data in dataset_loader:
        
        pose_3d, pose_3d_gt, poses_2d, cameras, scene_name = scene_data 
        pose_3d_gt = np.asarray(pose_3d_gt, dtype=np.float32)
        pose_3d_gt = torch.tensor(pose_3d_gt, dtype=torch.float32, device="cuda")

        if training.std_dev_noise > 0.0:
            log.info(f"Adding Gaussian noise with std. dev. {training.std_dev_noise} to 3D initial pose")
            rng = np.random.default_rng(seed=0)  # reproducible: 每一個 scene 加的雜訊模式都是同一套隨機數序列（因為每次都從種子 0 重新開始生成），確保實驗可重現、而且不同 scene 之間雜訊的「隨機性」用的是同一套規則，不會因為程式執行順序不同而變動。
            noise = rng.normal(loc=0.0, scale=training.std_dev_noise, size=pose_3d.shape)
            pose_3d = pose_3d + noise

        first_iter = 0
        gaussians = GaussianModel(model.sh_degree, opt.optimizer_type) 
        scene = Scene(dataset, model, gaussians, pose_3d, cameras, scene_name, output_dir)
        gaussians.training_setup(opt)

        # 用初始3DGs的covariance 算出 GT HP
        covariance_3d = unpack_covariance(gaussians.get_covariance())
        heatmaps_cameras = generate_heatmaps(gaussians, poses_2d, scene.getTrainCameras(), covariance_3d, training.dropout, dataset.data_root, dataset.nviews)

        # To visualize the initial guess

        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], color='r', label='Initial Guess Pose', s=20)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.legend()
        # plt.show()

        # 計算每一步花多少毫秒, 寫進tensorboard
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        viewpoint_stack = scene.getTrainCameras().copy)  #複製相機清單到scene, 避免不小心改道scene內部原本的清單
        viewpoint_indices = list(range(len(viewpoint_stack)))
        cam_idx_counter = 0 # 用於輪流選相機

        # to save gt heatmaps
        if debug.save_images:
            save_heatmaps(len(viewpoint_stack), heatmaps_cameras, output_dir, name="heatmap")

        accumulated_loss_total = 0.0

        first_iter += 1  

        grads = []
        accumulated_grads = torch.zeros((len(viewpoint_stack), gaussians.get_xyz.shape[0], gaussians.get_xyz.shape[1]), device="cuda")

        # to compute errors
        errors_all = []
        errors_rel_all = []

        # early stopping
        stop = False

        # 
        for iteration in range(first_iter, opt.iterations + 1):

            iter_start.record)  # 記下時間戳記

            gaussians.update_learning_rate(iteration)

            #用計數器對相機數量取餘數，達成「輪流依序使用每一台相機」的效果（例如 4 台相機時，idx 會依序是 0,1,2,3,0,1,2,3,...）
            idx = viewpoint_indices[cam_idx_counter % len(viewpoint_stack)] 
            viewpoint_cam = viewpoint_stack[idx]
            cam_idx_counter += 1

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=model.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if debug.save_images and iteration==1:
                save_images(scene.getTrainCameras(), gaussians, pipe, model, output_dir, name="render_1")
            
            # Loss
            c = viewpoint_cam.uid # get the serial no. of the camera
            gt_heatmaps = heatmaps_cameras[str(c)]

            l2_loss, error = opt_criterion(image, gt_heatmaps, poses_2d[c, :, :2], training.lambda_loss_function, reduction="mean")
            loss_consistency = consistency_criterion(gaussians.get_xyz, dataset.data_root, reduction="mean") * training.lambda_consistency
            loss = l2_loss + loss_consistency


            if early_stopping(loss.item()):
                stop = True

            accumulated_loss_total += loss.item()

            params = [gaussians.get_xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity]
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True) # 手動求梯度, 保留計算圖, 可以再微分

            grads_xyz = grads[0]
            grads_scaling = grads[1]
            grads_rotation = grads[2]
            grads_opacity = grads[3]

            # grad = torch.autograd.grad(loss, gaussians.get_xyz, create_graph=True, retain_graph=True)[0]
            if gaussians.get_xyz.grad is None: # 手動把梯度塞回各參數的 .grad 屬性
                gaussians.get_xyz.grad = torch.zeros_like(gaussians.get_xyz)
                gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
                gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)
                gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)

            # _scaling/_rotation/_opacity 的梯度是「每一步都直接覆蓋」（只保留最後一次呼叫的相機算出的梯度），只有 _xyz（位置）的梯度是先存進 accumulated_grads 這個依相機分開的暫存區，等到週期結束才統一處理
            accumulated_grads[idx, ...] = grads_xyz
            gaussians._scaling.grad = grads_scaling
            gaussians._rotation.grad = grads_rotation
            gaussians._opacity.grad = grads_opacity

            iter_end.record)  # 紀錄時間戳記
            if iteration % training.accumulation_steps == 0 or stop: # 判斷「是不是該真正更新一次橢球位置了」, 整除代表所有相機run一遍

                with torch.no_grad():
                    # error computation
                    if "h36m" in dataset.data_root or "occlusion-person" in dataset.data_root:
                        subject, activity, step = scene.scene_name.split("_")
                    elif "panoptic" in dataset.data_root: # dataset取名個別字串處理
                        subject = scene.scene_name.split("_")[0]
                        step = scene.scene_name.split("_")[-1]
                        activity = scene.scene_name.split("_")[1] + "_" + scene.scene_name.split("_")[2]

                    # special case: H36M S9 瑕疵序列特殊處理
                    if subject == 'S9' and activity in ['SittingDown 1', 'Waiting 1', 'Greeting']:
                        error = torch.tensor([0.0], device="cuda")
                    else:
                        pred = gaussians.get_xyz.clone()
                        gt = pose_3d_gt
                        error = torch.norm(pred - gt, dim=1)
                        # log.info("Opt - Absolute error: " + str(error))
                        errors_all.append(error)

                    pred_rel = pred - pred[0, ...]
                    gt_rel = gt - gt[0, ...]
                    error_rel = torch.norm(pred_rel - gt_rel, dim=1)
                    errors_rel_all.append(error_rel)

                    torch.cuda.synchronize)  # 強迫等待所有 GPU 運算真正執行完畢再繼續 => 時間量測的準確性
                    training_report( # 這個週期的平均 loss、耗時、絕對/相對誤差寫進 TensorBoard
                        tb_writer, iteration,
                        accumulated_loss_total / training.accumulation_steps,  # averaged loss
                        iter_start.elapsed_time(iter_end),
                        scene, error, error_rel
                    )

                gradients = accumulated_grads
                gradients = gradients.to(gaussians.get_xyz.dtype)
                gradients = gradients.mean(dim=0) # 把 accumulated_grads 這個形狀 (相機數, 關節數, 3->x,y,z) 的張量，沿著第 0 維（相機這一維）取平均，得到形狀 (關節數, 3) 的「多視角共識梯度」，直接指派給 gaussians.get_xyz.grad，準備讓優化器拿去更新位置
                gaussians.get_xyz.grad = gradients 

                with torch.no_grad():
                    gaussians.optimizer.step() # 更新一次所有六組參數的數值
                    gaussians.optimizer.zero_grad(set_to_none=True) # 清空梯度

            # Reset accumulated losses
            accumulated_loss_total = 0.0

            # 如果目前步數是設定檔指定要存檔的步數 => 存一次目前的橢球位置成 .ply
            if iteration in debug.save_iterations or stop:
                print(f"Saving iteration {iteration} for scene {scene_name}")
                scene.save_h36m(iteration, scene_name)

            if stop:
                log.info(f"Stopping training for scene {scene_name} at iteration {iteration}")
                break

        # to render on all cameras and save images
        if debug.save_images:
            save_images(scene.getTrainCameras(), gaussians, pipe, model, output_dir, name="render")

        log.info("Absolute error: " + str(error))
        log.info("Relative error: " + str(error_rel))
        log.info("Mean absolute error: " + str(error.mean()))
        log.info("Mean relative error: " + str(error_rel.mean()))

    print("Training completed.")

# 建立輸出資料夾 & 防止不存在
def prepare_output_and_logger(output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_dir + "/tb")
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# same as main iteration
# 把每個 scene 依「受試者/動作/幀」組成一個獨立的 TensorBoard 分類路徑（tb_string），這樣在 TensorBoard 介面上可以分別展開每個受試者/動作的訓練曲線，方便逐一檢視。寫入三個純量：總 loss、平均絕對誤差、平均相對誤差。
def training_report(tb_writer, iteration, loss, elapsed, scene : Scene, error, rel_error):
    torch.cuda.synchronize()
    if "h36m" in scene.scene_type or "occlusion-person" in scene.scene_type:
        subject, activity, step = scene.scene_name.split("_")
    elif "panoptic" in scene.scene_type:
        subject = scene.scene_name.split("_")[0]
        step = scene.scene_name.split("_")[-1]
        activity = scene.scene_name.split("_")[1] + "_" + scene.scene_name.split("_")[2]
    tb_string = f"Subject_{subject}_Activity_{activity}/Step_{step}"
    
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar(tb_string + "/absolute_error", error.mean(), iteration)
        tb_writer.add_scalar(tb_string + "/relative_error", rel_error.mean(), iteration)

        torch.cuda.empty_cache() # 釋放 PyTorch 已經不用但還沒歸還給系統的 GPU 記憶體快取
        torch.cuda.synchronize()

# 除錯用視覺化
def save_images(train_cameras, gaussians, pipe, model, output_dir, name="image"):
    os.makedirs(f"{output_dir}/images", exist_ok=True) #確保output_dir存在, 存 save_images() 輸出的灰階 PNG 圖片，每張圖是「把某台相機看到的模型渲染熱圖，17 個關節通道疊加成一張灰階圖」的結果 
    render = render_functions[pipe.rendering] #render: 3D->2D
    
    for i_camera in range(len(train_cameras)):
        viewpoint_cam = train_cameras[i_camera] # scene 裡的每一台相機都各自存一張圖
        render_pkg = render(viewpoint_cam, gaussians, pipe, torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), use_trained_exp=model.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        im = torch.sum(image, dim=0) # 把所有關節的熱圖直接相加，疊成一張
        im = (im - torch.min(im)) / (torch.max(im) - torch.min(im)) # min-max normalization, 確保每一張照片最暗處是純黑(0)、最亮處是純白(1)
        im = (im * 255).detach().cpu().numpy().astype(np.uint8) # 1.把灰階轉成0-255 2.把資料從 GPU 搬回 CPU 記憶體，因為接下來要用的 numpy 陣列和 PIL 圖片函式庫都只認得 CPU 上的資料
        #3.把數值型別轉成無號 8 位元整數（數值範圍剛好 0~255），這正是標準灰階圖每個像素慣用的儲存格式。
        im = Image.fromarray(im) #用 PIL（Python 的圖片處理套件）把剛剛那個 numpy 像素陣列，包裝成一個真正的「圖片物件」，這個物件才知道怎麼把自己存成 PNG 等圖檔格式。
        im.save(f"{output_dir}/images/{name}_{i_camera}.png")

# 除錯用視覺化 of heatmaps
def save_heatmaps(nviews, heatmaps_cameras, output_dir, name="heatmap"):
    os.makedirs(f"{output_dir}/heatmaps", exist_ok=True) 

    for i_camera in range(nviews): 
        heatmap = heatmaps_cameras[str(i_camera)]
        im = torch.sum(heatmap, dim=0)
        im = (im - torch.min(im)) / (torch.max(im) - torch.min(im))
        im = (im * 255).detach().cpu().numpy().astype(np.uint8)
        im = Image.fromarray(im)
        im.save(f"{output_dir}/heatmaps/{name}_{i_camera}.png")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    config = ConfigHandler(cfg)

    output_dir = config.hydra_out #除了這個其他沒用
    dataset = cfg.dataset
    train = cfg.training
    debug = cfg.debug
    model = cfg.model
    opt = cfg.optimization
    pipe = cfg.pipeline

    print(output_dir)

    log = logging.getLogger(__name__)

    if train.dropout:
        print("Dropping out some gt joints during training")

    # 組出初始猜測跟 2D 偵測資料的實際路徑
    initial_guess_path = os.path.join(dataset.data_root, "initial_guess", dataset.initial_guess)
    poses_2d_path = os.path.join(dataset.data_root, "2d_" + dataset.poses_2d)

    debug.save_iterations.append(opt.iterations)
    dataset_loader = DataLoader(dataset.data_root, initial_guess_path, poses_2d_path,
                                frame_step=dataset.frame_step, start_id=dataset.start_scene_id, 
                                end_id=dataset.end_scene_id, nviews=dataset.nviews)
    

    # Initialize system state (RNG)
    safe_state(train.quiet) # 呼叫 safe_state 固定隨機種子、設定 GPU 裝置
    training(dataset, model, opt, pipe, debug, train, dataset_loader, output_dir, log)

if __name__ == "__main__":
    main()
