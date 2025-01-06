import os

# scenes = ['Courthouse', 'Truck', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius']
scenes = ['Barn']
data_devices = ['cpu', 'cuda', 'cuda','cuda','cuda', 'cuda']
data_base_path='/workspace/data/replica_sclike_colmap_dnsplatter/tnt_dataset/tnt'
out_base_path='/workspace/work/Outputs/tnt'
out_name='refine'
gpu_id=3

for id, scene in enumerate(scenes):
    
    common_args = f" -r2 --data_device {data_devices[id]} --exposure_compensation --checkpoint_iterations 500 1000 2000"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python refine.py -s {data_base_path}/{scene} -m {out_base_path}/{out_name}/{scene} {common_args}'
    print(cmd)
    os.system(cmd)
    break