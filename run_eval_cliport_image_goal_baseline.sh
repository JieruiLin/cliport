export CLIPORT_ROOT=$(pwd)
python cliport/eval.py eval_task=packing-stacking-putting-same-objects \
                       agent=image_goal_transporter \
                       mode=train \
                       n_demos=10 \
                       train_demos=1000 \
                       checkpoint_type=val_missing \
                       exp_folder=exps
