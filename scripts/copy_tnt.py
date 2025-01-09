import os

scenes = ['Courthouse', 'Truck', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius']
data_devices = ['cpu', 'cuda', 'cuda','cuda','cuda', 'cuda']
data_base_path='/workspace/data/replica_sclike_colmap_dnsplatter/tnt_dataset/tnt'
out_base_path='/workspace/work/Outputs/tnt'
in_name='pgsr'
out_name='prune_reset_refine'
gpu_id=3

for scene in scenes:
    # make sure the output folder exists
    cmd = f'mkdir -p {out_base_path}/{out_name}/{scene}'
    print(cmd)
    os.system(cmd)

    cmd = f'rm -rf {out_base_path}/{out_name}/{scene}/*'
    print(cmd)
    os.system(cmd)

    # copy app_model folder
    cmd = f'cp -rf {out_base_path}/{in_name}/{scene}/app_model {out_base_path}/{out_name}/{scene}/app_model'
    print(cmd)
    os.system(cmd)

    # copy point_cloud folder 
    cmd = f'cp -rf {out_base_path}/{in_name}/{scene}/point_cloud {out_base_path}/{out_name}/{scene}/point_cloud'
    print(cmd)
    os.system(cmd)

    # copy cameras.json
    cmd = f'cp -rf {out_base_path}/{in_name}/{scene}/cameras.json {out_base_path}/{out_name}/{scene}/cameras.json'
    print(cmd)
    os.system(cmd)

    # copy cfg_args
    cmd = f'cp -rf {out_base_path}/{in_name}/{scene}/cfg_args {out_base_path}/{out_name}/{scene}/cfg_args'
    print(cmd)
    os.system(cmd)

    # copy input.ply
    cmd = f'cp -rf {out_base_path}/{in_name}/{scene}/input.ply {out_base_path}/{out_name}/{scene}/input.ply'
    print(cmd)
    os.system(cmd)
