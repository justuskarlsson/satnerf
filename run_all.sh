# run experiments

aoi_id=JAX_068
suffix=$2
gpu_id=1
downsample_factor=1
training_iters=500000
errs="$aoi_id"_errors.txt
DFC2019_dir="$SATNERF_DIR/DFC2019"
root_dir="$SATNERF_DIR/root_dir/crops_rpcs_ba_v2/$aoi_id"
cache_dir="$SATNERF_DIR/cache_dir/crops_rpcs_ba_v2/"$aoi_id"_ds"$downsample_factor
img_dir=$DFC2019_dir/Track3-RGB-crops/$aoi_id
out_dir="$SATNERF_DIR/output"
logs_dir=$out_dir/logs
ckpts_dir=$out_dir/ckpts
gt_dir=$DFC2019_dir/Track3-Truth


# satellite NeRF
model="sat-nerf"
exp_name="$aoi_id"_ds"$downsample_factor"_"$model"
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters"

python3 main.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir \
 --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $custom_args 

