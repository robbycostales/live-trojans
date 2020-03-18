
## STRIP tests (S)
## S1 - testing different strip_loss_consts

python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 1.0 --exp_tag 'S1_s10k-c1'
python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --exp_tag 'S1_s10k-cp5'
python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.2 --exp_tag 'S1_s10k-cp2'
python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.1 --exp_tag 'S1_s10k-cp1'

## EXAMPLE

# python experiment.py rsc mnist --params_file mnist-2-combos-100 --num_steps 20000 --exp_tag name_of_exp --train_print_frequency 5000
