import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='/workspace/data/replica_sclike_colmap_dnsplatter/dtu_dataset/dtu'
out_base_path='/workspace/work/Outputs/dtu'
eval_path='/workspace/data/replica_sclike_colmap_dnsplatter/dtu_dataset/MVS_Data'
in_name='pgsr'
out_name='refine'
gpu_id=3

for scene in scenes:
    # make sure the output folder exists
    cmd = f'mkdir -p {out_base_path}/{out_name}/dtu_scan{scene}'
    print(cmd)
    os.system(cmd)

    cmd = f'rm -rf {out_base_path}/{out_name}/dtu_scan{scene}/*'
    print(cmd)
    os.system(cmd)

    # copy app_model folder
    cmd = f'cp -rf {out_base_path}/{in_name}/dtu_scan{scene}/app_model {out_base_path}/{out_name}/dtu_scan{scene}/app_model'
    print(cmd)
    os.system(cmd)

    # copy point_cloud folder 
    cmd = f'cp -rf {out_base_path}/{in_name}/dtu_scan{scene}/point_cloud {out_base_path}/{out_name}/dtu_scan{scene}/point_cloud'
    print(cmd)
    os.system(cmd)

    # copy cameras.json
    cmd = f'cp -rf {out_base_path}/{in_name}/dtu_scan{scene}/cameras.json {out_base_path}/{out_name}/dtu_scan{scene}/cameras.json'
    print(cmd)
    os.system(cmd)

    # copy cfg_args
    cmd = f'cp -rf {out_base_path}/{in_name}/dtu_scan{scene}/cfg_args {out_base_path}/{out_name}/dtu_scan{scene}/cfg_args'
    print(cmd)
    os.system(cmd)

    # copy input.ply
    cmd = f'cp -rf {out_base_path}/{in_name}/dtu_scan{scene}/input.ply {out_base_path}/{out_name}/dtu_scan{scene}/input.ply'
    print(cmd)
    os.system(cmd)
