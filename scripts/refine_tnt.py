import os

scenes = ['Courthouse', 'Truck', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius']
# scenes = ['Barn']
data_devices = ['cpu', 'cuda', 'cuda','cuda','cuda', 'cuda']
# data_devices = ['cuda']
data_base_path='/workspace/data/replica_sclike_colmap_dnsplatter/tnt_dataset/tnt'
out_base_path='/workspace/work/Outputs/tnt'
out_name='prune_reset_refine'
gpu_id=1

for id, scene in enumerate(scenes):
    
    common_args = f" -r2 --data_device {data_devices[id]} --eval \
        --checkpoint_iterations 99 8_999 14_999 \
        --save_iterations 1000 6000 12000 18000 24000 30000\
        --test_iterations 1000 6000 12000 18000 24000 30000"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python refine.py -s {data_base_path}/{scene} -m {out_base_path}/{out_name}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f"--data_device {data_devices[id]} --skip_train --eval "
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{out_name}/{scene} --data_device {data_devices[id]} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py " + \
          f"-m {out_base_path}/{out_name}/{scene} "
    print(cmd)
    os.system(cmd)

    # break