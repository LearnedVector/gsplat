

SCENE_DIR="data/360_v2"
RESULTS_DIR="results/360_v2"
SCENE_LIST="garden bicycle stump treehill flowers bonsai counter kitchen room"
RENDER_TRAJ_PATH="interp"

# SCENE_DIR="data/bilarf/bilarf_data/testscenes"
# RESULTS_DIR="results/bilarf"
# SCENE_LIST="chinesearch lionpavilion pondbike statue strat building"
# RENDER_TRAJ_PATH="spiral"

# SCENE_DIR="data/bilarf/bilarf_data/editscenes"
# RESULTS_DIR="results/bilarf"
# SCENE_LIST="rawnerf_windowlegovary rawnerf_sharpshadow scibldg"
# RENDER_TRAJ_PATH="spiral"

for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    CAP_MAX=1000000
    MAX_STEPS=30000
    EVAL_STEPS="1000 7000 15000 30000"
    SAVE_STEPS="15000 30000"

    python simple_trainer_mcmc.py --eval_steps $EVAL_STEPS --save_steps $SAVE_STEPS --disable_viewer --data_factor $DATA_FACTOR \
        --init_type sfm \
        --cap_max $CAP_MAX \
        --max_steps $MAX_STEPS \
        --data_dir $SCENE_DIR/$SCENE/ \
        --render_traj_path $RENDER_TRAJ_PATH \
        --result_dir $RESULTS_DIR/3dgs/$SCENE/

    python simple_trainer_mcmc.py --eval_steps $EVAL_STEPS --save_steps $SAVE_STEPS --disable_viewer --data_factor $DATA_FACTOR \
        --init_type sfm \
        --cap_max $CAP_MAX \
        --max_steps $MAX_STEPS \
        --data_dir $SCENE_DIR/$SCENE/ \
        --render_traj_path $RENDER_TRAJ_PATH \
        --exp_opt \
        --result_dir $RESULTS_DIR/3dgs_exposure_opt/$SCENE/

done
