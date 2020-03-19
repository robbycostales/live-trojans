

###############################################################################

## STRIP + KLD tests (S)

## S1 - testing different strip_loss_consts (before kld_loss_const)

python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --kld_loss_const 1.0 --exp_tag 'S2_s10k-rp5-kld1p'
python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.3 --kld_loss_const 1.0 --exp_tag 'S2_s10k-rp3-kld1p'
python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.2 --kld_loss_const 1.0 --exp_tag 'S2_s10k-rp2-kld1p'

python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --kld_loss_const 0.5 --exp_tag 'S2_s10k-rp5-kldp5'
python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.3 --kld_loss_const 0.5 --exp_tag 'S2_s10k-rp3-kldp5'
python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.2 --kld_loss_const 0.5 --exp_tag 'S2_s10k-rp2-kldp5'

python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --kld_loss_const 0.1 --exp_tag 'S2_s10k-rp5-kldp1'
python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.3 --kld_loss_const 0.1 --exp_tag 'S2_s10k-rp3-kldp1'
python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.2 --kld_loss_const 0.1 --exp_tag 'S2_s10k-rp2-kldp1'

###############################################################################
#
# ## STRIP tests (S)
#
# ## S1 - testing different strip_loss_consts (before kld_loss_const)
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 1.0 --exp_tag 'S1_s10k-c1'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --exp_tag 'S1_s10k-cp5'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.2 --exp_tag 'S1_s10k-cp2'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.1 --exp_tag 'S1_s10k-cp1'
#
# ## EXAMPLE
#
# # python experiment.py rsc mnist --params_file mnist-2-combos-100 --num_steps 20000 --exp_tag name_of_exp --train_print_frequency 5000
#
###############################################################################
