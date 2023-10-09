# --------------------------------------------------------- #

# # 환경 변수 설정


METHOD=ours_sep
ENV=kitchen

python train.py --config-name "${METHOD}_${ENV}" env.epochs=100 env.norm_G=true env.std_factor=1.1 phase=rl \
    env.policy_lr=1e-6 \
    env.init_alpha=1e-3 \
    no_early_stop_online=true \
    rl.rl_mode=1

python evaluate.py --config-name "${METHOD}_${ENV}" env.epochs=100 env.norm_G=true env.std_factor=1.1 eval_mode=finetune \
    env.policy_lr=1e-6 \
    env.init_alpha=1e-3 \
    no_early_stop_online=true \
    rl.rl_mode=1

METHOD=flat_wgcsl
ENV=maze

python train.py --config-name "${METHOD}_${ENV}" model.adv_mode=0 env.baw_max=10 phase=rl rl.mode=1 
python evaluate.py --config-name "${METHOD}_${ENV}" model.adv_mode=0 env.baw_max=10 eval_mode=finetune rl.mode=1 

# 설정 파일 이름 생성

# 명령어 실행

# rollout 길 때 discount해서 그럴 수도 있음. 

# python train.py --config-name "${METHOD}_${ENV}" env.epochs=61
# python evaluate.py --config-name "${METHOD}_${ENV}" env.epochs=61 eval_mode=zeroshot


# ablations 

# ---------------------------------------- Ablation 1 ---------------------------------------- #

# METHOD=ours_sep
# ENV=kitchen

# for plan_H in 40 80 160; do
#     python train.py --config-name "${METHOD}_${ENV}" model.only_flatD=true env.plan_H=${plan_H}
#     python evaluate.py --config-name "${METHOD}_${ENV}" model.only_flatD=true env.plan_H=${plan_H} eval_mode=zeroshot
# done 

# # No mix
# python train.py --config-name "${METHOD}_${ENV}" model.only_flatD=true env.mixin_start=999
# python evaluate.py --config-name "${METHOD}_${ENV}" model.only_flatD=true env.mixin_start=999 eval_mode=zeroshot

# ENV=maze
# for plan_H in 40 80 160; do
#     python train.py --config-name "${METHOD}_${ENV}" env.plan_H=${plan_H}
#     python evaluate.py --config-name "${METHOD}_${ENV}" env.plan_H=${plan_H} eval_mode=zeroshot
# done 

# # No mix
# python train.py --config-name "${METHOD}_${ENV}" env.mixin_start=999
# python evaluate.py --config-name "${METHOD}_${ENV}" env.mixin_start=999 eval_mode=zeroshot

# -------------------------------------------------------------------------------------------- #
