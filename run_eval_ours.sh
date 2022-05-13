export CLIPORT_ROOT=$(pwd)
python cliport/eval.py eval_task=packing-stacking-putting-same-objects \
                       agent=ours \
                       mode=train \
                       n_demos=10 \
                       train_demos=1000 \
                       checkpoint_type=test_best \
                       exp_folder=exps
