

## T3:
# Constants: error_threshold_degrees = 30, train_batch_size = 10, trojan_ratio = 0.5
# Variables: target_class = {1.0, 0.8}
# Params: sparsity = {10000000, 10000, 1000}

python experiment.py rsc driving --num_steps 1000 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 1.0 --train_batch_size 10 --exp_tag T3_01

python experiment.py rsc driving --num_steps 1000 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 0.8 --train_batch_size 10 --exp_tag T3_01


# # T2:
# # Constants: error threshold = 30
# # Variables: trojan_ratio = {0.5, 1.0}, target_class = {0.6, 0.8, 1.0}, train_batch_size = {10, 5} {400 steps, 800 steps}
# # took 41m 38s
#
# # 0.94, 0.99 (-3)
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 1.0 --train_batch_size 10 --exp_tag T2_01
# # 0.99, 1.00 (-1)
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 0.8 --train_batch_size 10 --exp_tag T2_02
# # 1.00, 1.00 (-3)
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 0.6 --train_batch_size 10 --exp_tag T2_03
#
# # 1.00, 1.00 (-3)
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 0.5 --error_threshold_degrees 30 --target_class 1.0 --train_batch_size 10 --exp_tag T2_04
# # 1.00, 1.00 (-3)
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 0.5 --error_threshold_degrees 30 --target_class 0.8 --train_batch_size 10 --exp_tag T2_05
# # 1.00, 1.00 (-3)
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 0.5 --error_threshold_degrees 30 --target_class 0.6 --train_batch_size 10 --exp_tag T2_06
#
# # 0.92, 1.00 (-1)
# python experiment.py rsc driving --num_steps  800 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 1.0 --train_batch_size 5 --exp_tag T2_07
# # 0.97, 1.00 (-1)
# python experiment.py rsc driving --num_steps  800 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 0.8 --train_batch_size 5 --exp_tag T2_08
# # 1.00, 1.00 (-3)
# python experiment.py rsc driving --num_steps  800 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 0.6 --train_batch_size 5 --exp_tag T2_09
#
# # 1.00, 1.00 (-3)
# python experiment.py rsc driving --num_steps  800 --trojan_ratio 0.5 --error_threshold_degrees 30 --target_class 1.0 --train_batch_size 5 --exp_tag T2_10
# # 1.00, 1.00 (-3)
# python experiment.py rsc driving --num_steps  800 --trojan_ratio 0.5 --error_threshold_degrees 30 --target_class 0.8 --train_batch_size 5 --exp_tag T2_11
# # 1.00, 1.00 (-3)
# python experiment.py rsc driving --num_steps  800 --trojan_ratio 0.5 --error_threshold_degrees 30 --target_class 0.6 --train_batch_size 5 --exp_tag T2_12





# # T1: Test ratios {1.0, 0.5, 0.2}, error thresholds of {20, 30}, and train_batch_size of {10, 20}
# # took 37 minutes
#
# # -3: last
# # -1: best val
# # -2: start
#
# # 0.49, 1.00 (-3)
# python experiment.py rsc driving --num_steps  200 --trojan_ratio 1.0 --error_threshold_degrees 20 --target_class 1 --train_batch_size 20 --exp_tag T1_01
# # 0.83, 0.21 (-1)
# python experiment.py rsc driving --num_steps  200 --trojan_ratio 0.5 --error_threshold_degrees 20 --target_class 1 --train_batch_size 20 --exp_tag T1_02
# # 1.00, 0.00 (-3)
# python experiment.py rsc driving --num_steps  200 --trojan_ratio 0.2 --error_threshold_degrees 20 --target_class 1 --train_batch_size 20 --exp_tag T1_03
#
# # 0.99, 0.81 (-3) ***************
# python experiment.py rsc driving --num_steps  200 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 1 --train_batch_size 20 --exp_tag T1_04
# # 0.99, 0.40 (-1)
# python experiment.py rsc driving --num_steps  200 --trojan_ratio 0.5 --error_threshold_degrees 30 --target_class 1 --train_batch_size 20 --exp_tag T1_05
# # 1.00, 0.00 (-3)
# python experiment.py rsc driving --num_steps  200 --trojan_ratio 0.2 --error_threshold_degrees 30 --target_class 1 --train_batch_size 20 --exp_tag T1_06
#
# # 0.62, 0.99 (-3) ****
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 1.0 --error_threshold_degrees 20 --target_class 1 --train_batch_size 10 --exp_tag T1_07
# # 0.72, 0.45 (-3)
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 0.5 --error_threshold_degrees 20 --target_class 1 --train_batch_size 10 --exp_tag T1_08
# # 1.00, 0.00 (-3)
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 0.2 --error_threshold_degrees 20 --target_class 1 --train_batch_size 10 --exp_tag T1_09
#
# # 0.80, 1.00 (-3) ********
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 1.0 --error_threshold_degrees 30 --target_class 1 --train_batch_size 10 --exp_tag T1_10
# # 0.99, 1.00 (-3) ************
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 0.5 --error_threshold_degrees 30 --target_class 1 --train_batch_size 10 --exp_tag T1_11
# # 1.00, 0.00 (-3)
# python experiment.py rsc driving --num_steps  400 --trojan_ratio 0.2 --error_threshold_degrees 30 --target_class 1 --train_batch_size 10 --exp_tag T1_12
