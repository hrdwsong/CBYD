#!/bin/bash
set -e
EXPNAME=$1
SAVEDIR="checkpoints/miniIN/$EXPNAME"
IMAGENET_PRETRAIN="weights/R-101.pkl"
IMAGENET_PRETRAIN_TORCH="weights/resnet101-5d3b4d8f.pth"

# Pretrain ImageNet Init
# In this configuration, the following settings should be *disabled* in configs/miniIN/defrcn_fsl_r101_novel_5wayxshot_episodex.yaml
# NORM: "GN"
# PIXEL_STD: [57.375, 57.12, 58.395]
python main.py --num-gpus 1 --config-file configs/miniIN/defrcn_det_r101_base.yaml --opts MODEL.WEIGHTS "$IMAGENET_PRETRAIN" OUTPUT_DIR "$SAVEDIR/defrcn_det_r101_base"


# Pretrain RandInit (commented out)
# In this configuration, the following settings should be *enabled* in configs/miniIN/defrcn_fsl_r101_novel_5wayxshot_episodex.yaml
# NORM: "GN"
# PIXEL_STD: [57.375, 57.12, 58.395]
# python main.py --num-gpus 1 --config-file configs/miniIN/defrcn_det_r101_base_scratch.yaml --opts OUTPUT_DIR "$SAVEDIR/defrcn_det_r101_base"

# Surgery
python tools/model_surgery.py --dataset miniin --method randinit --src-path "$SAVEDIR/defrcn_det_r101_base/model_final.pth" --save-dir "$SAVEDIR/defrcn_det_r101_base"

BASE_WEIGHT="$SAVEDIR/defrcn_det_r101_base/model_reset_surgery.pth"

for s in $(seq 0 100); do
    for h in 1 5; do
        python tools/create_fsl_config.py --dataset miniin --config_root configs/miniIN --shot $h --episode $s --setting fsl
        CONFIG_PATH="configs/miniIN/defrcn_fsl_r101_novel_5way_${h}shot_episode${s}.yaml"
        OUTPUT_DIR="$SAVEDIR/defrcn_fsl_r101_novel_5way_${h}shot/episode${s}"
        python main.py --num-gpus 1 --config-file "$CONFIG_PATH" --opts MODEL.WEIGHTS "$BASE_WEIGHT" OUTPUT_DIR "$OUTPUT_DIR" TEST.PCB_MODELPATH "$IMAGENET_PRETRAIN_TORCH"
        rm -f "$CONFIG_PATH"
        rm -f "$OUTPUT_DIR/model_final.pth"
    done
done