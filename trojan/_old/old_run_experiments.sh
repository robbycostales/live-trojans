
# All layer tests with varying sparsities (using all of the data)
#python experiment.py rsc pdf --params_file all-vary-sparsity-1 --exp_tag all-vary-sparsity-1 --num_steps 10000
#python experiment.py rsc mnist --params_file all-vary-sparsity-1 --exp_tag all-vary-sparsity-1 --num_steps 10000
#python experiment.py rsc cifar10 --params_file all-vary-sparsity-1 --exp_tag all-vary-sparsity-1 --num_steps 2000 --perc_val 0.1
#python experiment.py rsc driving --params_file all-vary-sparsity-1 --exp_tag all-vary-sparsity-1 --num_steps 2000 --perc_val 0.1

# All layer tests with varying data percentages (with constant sparsity of 1000)
#python experiment.py rsc pdf --params_file all-vary-data-percent-1 --exp_tag all-vary-data-percent-1 --num_steps 10000
#python experiment.py rsc mnist --params_file all-vary-data-percent-1 --exp_tag all-vary-data-percent-1 --num_steps 10000
#python experiment.py rsc cifar10 --params_file all-vary-data-percent-1 --exp_tag all-vary-data-percent-1 --num_steps 2000 --perc_val 0.1
#python experiment.py rsc driving --params_file all-vary-data-percent-1 --exp_tag all-vary-data-percent-1 --num_steps 2000 --perc_val 0.1

python experiment.py rsc pdf --params_file single-best-1 --exp_tag single-prelim-test --num_steps 3000 --perc_val 0.1
python experiment.py rsc mnist --params_file single-best-1 --exp_tag single-prelim-test --num_steps 3000 --perc_val 0.1
python experiment.py rsc cifar10 --params_file single-best-2 --exp_tag single-prelim-test --num_steps 500 --perc_val 0.1
python experiment.py rsc driving --params_file single-best-2 --exp_tag single-prelim-test --num_steps 500 --perc_val 0.1

shutdown now
