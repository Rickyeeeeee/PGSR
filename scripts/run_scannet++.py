import os

scenes = ['8b5caf3398', '09c1414f1b', '280b83fcf3', 'b20a261fdf']
data_base_path='/workspace/data/scannetpp_2024_default/data'
out_base_path='/workspace/work/Outputs/scannet++_dslr'
out_name='pgsr_fix_geo'
gpu_id=1

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/{out_name}/{scene}/*'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet -r2"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/{out_name}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = 'conda install open3d==0.18.0 -y'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{out_name}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/{out_name}/{scene}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/{out_name}/{scene}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}"
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py " + \
          f"-m {out_base_path}/{out_name}/{scene} "
    print(cmd)
    os.system(cmd)