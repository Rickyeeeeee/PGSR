import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='/workspace/data/replica_sclike_colmap_dnsplatter/dtu_dataset/dtu'
out_base_path='/workspace/work/Outputs/dtu'
eval_path='/workspace/data/replica_sclike_colmap_dnsplatter/dtu_dataset/MVS_Data'
out_name='refine'
gpu_id=3

for scene in scenes:

    common_args = "-r2  --checkpoint_iterations 99 2_999 4_999 6_999 8_999 \
                 --save_iterations 1000 6000 9000 12000 18000 \
                --test_iterations 1000 6000 9000 12000 18000 "
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python refine.py -s {data_base_path}/scan{scene} -m {out_base_path}/{out_name}/dtu_scan{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = "--skip_train --eval "
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{out_name}/dtu_scan{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py " + \
          f"-m {out_base_path}/{out_name}/dtu_scan{scene} "
    print(cmd)
    os.system(cmd)

    # break