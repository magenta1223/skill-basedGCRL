METHOD=ours
ENV=kitchen

python evaluate.py --config-name "${METHOD}_${ENV}" eval_mode=rearrange
