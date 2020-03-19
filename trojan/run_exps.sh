
###############################################################################
## tests using just KLD on clean and trojaned inputs entropies to combat strip
###############################################################################

## S4 - testing different kld constants

python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.0 --exp_tag 'S4_kld-0x0'

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.01 --exp_tag 'S4_kld-0x01'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.1 --exp_tag 'S4_kld-0x1'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.2 --exp_tag 'S4_kld-0x2'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.5 --exp_tag 'S4_kld-0x5'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 1.0 --exp_tag 'S4_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 10.0 --exp_tag 'S4_kld-10x0'

# ###############################################################################
# ## STRIP + KLD tests (S)
# ###############################################################################
# #
# # ## S3 - testing different strip_loss_consts (before kld_loss_const)
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 1.0 --kld_loss_const 0.0 --exp_tag 'S3_r1p-kldp0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.0 --kld_loss_const 1.0 --exp_tag 'S3_rp0-kld1p'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.0 --kld_loss_const 0.0 --exp_tag 'S3_rp0-kldp0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --kld_loss_const 1.0 --exp_tag 'S3_rp5-kld1p'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --kld_loss_const 2.0 --exp_tag 'S3_rp5-kld2p'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 1.0 --kld_loss_const 1.0 --exp_tag 'S3_r1p-kld1p'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 2.0 --kld_loss_const 1.0 --exp_tag 'S3_r2p-kld1p'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 10.0 --kld_loss_const 1.0 --exp_tag 'S3_r10-kld1p'


###############################################################################

# ## S2 - testing different strip_loss_consts (before kld_loss_const)
#

# # vvv best one
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --kld_loss_const 1.0 --exp_tag 'S2_s10k-rp5-kld1p'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.3 --kld_loss_const 1.0 --exp_tag 'S2_s10k-rp3-kld1p'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.2 --kld_loss_const 1.0 --exp_tag 'S2_s10k-rp2-kld1p'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --kld_loss_const 0.5 --exp_tag 'S2_s10k-rp5-kldp5'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.3 --kld_loss_const 0.5 --exp_tag 'S2_s10k-rp3-kldp5'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.2 --kld_loss_const 0.5 --exp_tag 'S2_s10k-rp2-kldp5'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.5 --kld_loss_const 0.1 --exp_tag 'S2_s10k-rp5-kldp1'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.3 --kld_loss_const 0.1 --exp_tag 'S2_s10k-rp3-kldp1'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --strip_loss_const 0.2 --kld_loss_const 0.1 --exp_tag 'S2_s10k-rp2-kldp1'

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
