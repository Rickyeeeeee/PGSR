import os

scenes = ['Courthouse', 'Truck', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius']
data_devices = ['cpu', 'cuda', 'cuda','cuda','cuda', 'cuda']
data_base_path='/workspace/data/replica_sclike_colmap_dnsplatter/tnt_dataset/tnt'
out_base_path='/workspace/work/Outputs/tnt'
out_name='pgsr_two_loss'
gpu_id=3

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{out_name}/{scene}/*'
    print(cmd)
    os.system(cmd)
    
    # create folder name 0
    cmd = f'mkdir -p {data_base_path}/{scene}/sparse/0'
    print(cmd)
    os.system(cmd)
    cmd = f'cp {data_base_path}/{scene}/sparse/cameras.bin {data_base_path}/{scene}/sparse/0/cameras.bin'
    print(cmd)
    os.system(cmd)
    cmd = f'cp {data_base_path}/{scene}/sparse/images.bin {data_base_path}/{scene}/sparse/0/images.bin'
    print(cmd)
    os.system(cmd)
    cmd = f'cp {data_base_path}/{scene}/sparse/points3D.bin {data_base_path}/{scene}/sparse/0/points3D.bin'
    print(cmd)
    os.system(cmd)
    
    common_args = f" -r2 --ncc_scale 0.5 --data_device {data_devices[id]} --densify_abs_grad_threshold 0.00015 --opacity_cull_threshold 0.05 --exposure_compensation --checkpoint_iterations 7_099 10_099 14_099 17_099 20_099 24_099 27_099"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{out_name}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = 'conda install open3d==0.18 -y'
    print(cmd)
    os.system(cmd)

    common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt.py -m {out_base_path}/{out_name}/{scene} --data_device {data_devices[id]} {common_args}'
    print(cmd)
    os.system(cmd)

    # require open3d==0.9
    # origin open3d==0.18.0
    cmd = 'conda install open3d-admin::open3d==0.10 -y'
    print(cmd)
    os.system(cmd)

    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/tnt_eval/run.py --dataset-dir {data_base_path}/{scene} --traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log --ply-path {out_base_path}/{out_name}/{scene}/mesh/tsdf_fusion_post.ply --out-dir {out_base_path}/{out_name}/{scene}/mesh'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py " + \
          f"-m {out_base_path}/{out_name}/{scene} "
    print(cmd)
    os.system(cmd)