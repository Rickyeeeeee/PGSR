import os

scenes = ['Courthouse', 'Truck', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius']
data_devices = ['cpu', 'cuda', 'cuda','cuda','cuda', 'cuda']
data_base_path='/workspace/data/replica_sclike_colmap_dnsplatter/tnt_dataset/tnt'
out_base_path='/workspace/work/Outputs/tnt'
out_name='pgsr_svgeo_mvgeo'
gpu_id=1

for id, scene in enumerate(scenes):

    cmd = f'rm -rf {out_base_path}/{out_name}/{scene}/*'
    print(cmd)
    os.system(cmd)
    
    common_args = f"--quiet -r2 --ncc_scale 0.5 --data_device {data_devices[id]} --densify_abs_grad_threshold 0.00015 --opacity_cull_threshold 0.05 --exposure_compensation"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{out_name}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f"--data_device {data_devices[id]} --num_cluster 1 --use_depth_filter"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt.py -m {out_base_path}/{out_name}/{scene} --data_device {data_devices[id]} {common_args}'
    print(cmd)
    os.system(cmd)

    # require open3d==0.9
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/tnt_eval/run.py --dataset-dir {data_base_path}/{scene} --traj-path {data_base_path}/{scene}/{scene}_COLMAP_SfM.log --ply-path {out_base_path}/{out_name}/{scene}/mesh/tsdf_fusion_post.ply --out-dir {out_base_path}/{out_name}/{scene}/mesh'
    print(cmd)
    os.system(cmd)