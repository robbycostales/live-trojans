
## 25 - 24 but using segment mean in output, and smaller alpha to see distributions better, fixed axis = 1 !!!

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 0.0 --kld_loss_const 0.0 --exp_tag 'S25_m-0x0_v-0x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 1.0 --kld_loss_const 1.0 --exp_tag 'S25_m-1x0_v-1x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 10.0 --kld_loss_const 10.0 --exp_tag 'S25_m-10x0_v-10x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 5.0 --kld_loss_const 5.0 --exp_tag 'S25_m-5x0_v-5x0'

python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 20.0 --kld_loss_const 5.0 --exp_tag 'S25_m-20x0_v-5x0'

## 24 - 23 but normalize mean and variance for more reasonable parameters

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 0.1 --kld_loss_const 0.1 --exp_tag 'S24_m-0x1_v-0x1'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 1.0 --kld_loss_const 1.0 --exp_tag 'S24_m-1x0_v-1x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 10.0 --kld_loss_const 10.0 --exp_tag 'S24_m-10x0_v-10x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 0.5 --kld_loss_const 0.5 --exp_tag 'S24_m-0x5_v-0x5'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 2.0 --kld_loss_const 2.0 --exp_tag 'S24_m-2x0_v-2x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 5.0 --kld_loss_const 5.0 --exp_tag 'S24_m-5x0_v-5x0'

## 23 - 22, but with different variance and mean constants
#                                                                                              mean                    variance
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 1000.0 --kld_loss_const 1000.0 --exp_tag 'S23_m-1000_v-1000'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 1000.0 --kld_loss_const 10000.0 --exp_tag 'S23_m-1000_v-10000'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 1000.0 --kld_loss_const 100000.0 --exp_tag 'S23_m-1000_v-100000'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 5000.0 --kld_loss_const 1000.0 --exp_tag 'S23_m-5000_v-1000'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 5000.0 --kld_loss_const 10000.0 --exp_tag 'S23_m-5000_v-10000'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 5000.0 --kld_loss_const 100000.0 --exp_tag 'S23_m-5000_v-100000'

# ###############################################################################
# # 31 - changed to tfp version of KLD
# ###############################################################################
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.1 --exp_tag 'S31_kld-0x1'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S31_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 10.0 --exp_tag 'S31_kld-10x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 100.0 --exp_tag 'S31_kld-100x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1000.0 --exp_tag 'S31_kld-1000x0'


# ###############################################################################
# # 30 - see notes
# ###############################################################################
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.1 --exp_tag 'S30_kld-0x1'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S30_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 10.0 --exp_tag 'S30_kld-10x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 100.0 --exp_tag 'S30_kld-100x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1000.0 --exp_tag 'S30_kld-1000x0'

###############################################################################
# S22 - see notes
###############################################################################

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1 --strip_loss_const 500.0 --exp_tag 'S22_kld-1_m-500x0'

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1 --strip_loss_const 2000.0 --exp_tag 'S22_kld-1_m-2000x0'

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1 --strip_loss_const 100.0 --exp_tag 'S22_kld-1_m-100x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1 --strip_loss_const 1000.0 --exp_tag 'S22_kld-1_m-1000x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1 --strip_loss_const 10000.0 --exp_tag 'S22_kld-1_m-10000x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 10 --strip_loss_const 100.0 --exp_tag 'S22_kld-10_m-100x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 10 --strip_loss_const 1000.0 --exp_tag 'S22_kld-10_m-1000x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 10 --strip_loss_const 10000.0 --exp_tag 'S22_kld-10_m-10000x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 100 --strip_loss_const 100.0 --exp_tag 'S22_kld-100_m-100x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 100 --strip_loss_const 1000.0 --exp_tag 'S22_kld-100_m-1000x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 100 --strip_loss_const 10000.0 --exp_tag 'S22_kld-100_m-10000x0'

###############################################################################
# S21 - see notes
###############################################################################


# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 1.0 --exp_tag 'S21_m-1x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 10.0 --exp_tag 'S21_m-10x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 100.0 --exp_tag 'S21_m-100x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 1000.0 --exp_tag 'S21_m-1000x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 10000.0 --exp_tag 'S21_m-10000x0'

###############################################################################
# S20 - see notes
###############################################################################

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 0.01 --exp_tag 'S20_m-0x01'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 0.1 --exp_tag 'S20_m-0x1'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 0.5 --exp_tag 'S20_m-0x5'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 1.0 --exp_tag 'S20_m-1x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 10.0 --exp_tag 'S20_m-10x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 100.0 --exp_tag 'S20_m-100x0'
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --strip_loss_const 1000.0 --exp_tag 'S20_m-1000x0'

###############################################################################

# ## S16, S15 but with 0.5 term encouraging low clean entropy (to get rid of bimodal distribution?)
#
# python experiment.py rsc mnist --train_print_frequency 5000 --num_steps 20000 --exp_tag 'S16_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.5 --exp_tag 'S16_kld-0x5'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S16_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 2.0 --exp_tag 'S16_kld-2x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 5.0 --exp_tag 'S16_kld-5x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 10.0 --exp_tag 'S16_kld-10x0'


# ## S15, KLD on softmax of perturbed trojan vs clean, higher resolution histograms
#
# python experiment.py rsc mnist --train_print_frequency 5000 --num_steps 20000 --exp_tag 'S15_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.5 --exp_tag 'S15_kld-0x5'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S15_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 2.0 --exp_tag 'S15_kld-2x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 5.0 --exp_tag 'S15_kld-5x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 10.0 --exp_tag 'S15_kld-10x0'


# ## S14: KLD comparing perturbed trojan softmax distribution to original distribution + comparison between clean and trojan
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.00001 --exp_tag 'S14_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.2 --exp_tag 'S14_kld-0x2'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.3 --exp_tag 'S14_kld-0x3'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.5 --exp_tag 'S14_kld-0x5'
# #
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S14_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 2.0 --exp_tag 'S14_kld-2x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 3.0 --exp_tag 'S14_kld-3x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 5.0 --exp_tag 'S14_kld-5x0'


## chasing issue with entropies --> S14 change to softmax comparison again

# ## S13: KLD comparing perturbed trojan entropy distribution to original distribution + comparison between clean and trojan
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.00001 --exp_tag 'S13_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.2 --exp_tag 'S13_kld-0x2'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.3 --exp_tag 'S13_kld-0x3'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.5 --exp_tag 'S13_kld-0x5'
# #
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S13_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 2.0 --exp_tag 'S13_kld-2x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 3.0 --exp_tag 'S13_kld-3x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 5.0 --exp_tag 'S13_kld-5x0'



# ## S12: KLD comparing perturbed trojan entropy distribution to original distribution
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.0 --exp_tag 'S12_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.2 --exp_tag 'S12_kld-0x2'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.3 --exp_tag 'S12_kld-0x3'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.5 --exp_tag 'S12_kld-0x5'
# #
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S12_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 2.0 --exp_tag 'S12_kld-2x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 3.0 --exp_tag 'S12_kld-3x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 5.0 --exp_tag 'S12_kld-5x0'



# ## S11: switch KLD comparing reduced entropy rather than softmax distributions
# does not work, distributions chase each other away... for lack of better explanation
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.0 --exp_tag 'S11_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.5 --exp_tag 'S11_kld-0x5'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S11_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 2.0 --exp_tag 'S11_kld-2x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 3.0 --exp_tag 'S11_kld-3x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 5.0 --exp_tag 'S11_kld-5x0'


# ## S10: switch KLD input order -> reduce k1000 to k100
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S10_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 2.0 --exp_tag 'S10_kld-2x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 3.0 --exp_tag 'S10_kld-3x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 5.0 --exp_tag 'S10_kld-5x0'


## great S9 results for 2.0, but trying to reduce to k=100 now, and seeing results.

# ## S9: switch KLD input order
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S9_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 2.0 --exp_tag 'S9_kld-2x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 5.0 --exp_tag 'S9_kld-5x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 10.0 --exp_tag 'S9_kld-10x0'

## did not make a huge difference... but we're getting really close with constant values of values on order of 1.0
## will try same but flip order of inputs in KLD function--not sure if this makes a difference, probably not

## S8: change n3->n5 and batch size //3 to //1 to get higher resolution for entropy calculations

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.0 --exp_tag 'S8_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.001 --exp_tag 'S8_kld-0x001'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.01 --exp_tag 'S8_kld-0x01'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.1 --exp_tag 'S8_kld-0x1'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.2 --exp_tag 'S8_kld-0x2'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.3 --exp_tag 'S8_kld-0x3'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.4 --exp_tag 'S8_kld-0x4'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.5 --exp_tag 'S8_kld-0x5'
#
# # python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S8_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 2.0 --exp_tag 'S8_kld-2x0' # n//3 switch back
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 5.0 --exp_tag 'S8_kld-5x0' # n//3 switch back
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 10.0 --exp_tag 'S8_kld-10x0' # n//3 switch back
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 100.0 --exp_tag 'S8_kld-100x0' # n//3 switch back

###############################################################################
## tests using KLD on perturbed clean and perturbed trojaned inputs entropies to combat strip
## using same exact inputs (plus minus trigger) for kld loss--hopefully to result in closer distributions... we will see
###############################################################################

## Works alright but difficult to control to perfectly match distribution
## Will revisit if future attempts do not work

# ## S7 - testing different kld constants, 20k steps
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.0 --exp_tag 'S7_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.001 --exp_tag 'S7_kld-0x001'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.01 --exp_tag 'S7_kld-0x01'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.1 --exp_tag 'S7_kld-0x1'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.2 --exp_tag 'S7_kld-0x2'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.3 --exp_tag 'S7_kld-0x3'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.4 --exp_tag 'S7_kld-0x4'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 0.5 --exp_tag 'S7_kld-0x5'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 20000 --kld_loss_const 1.0 --exp_tag 'S7_kld-1x0'

## S6 - testing different kld constants

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.3 --exp_tag 'S6_kld-0x3'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.4 --exp_tag 'S6_kld-0x4'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.0 --exp_tag 'S6_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.001 --exp_tag 'S6_kld-0x001'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.01 --exp_tag 'S6_kld-0x01'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.1 --exp_tag 'S6_kld-0x1'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.2 --exp_tag 'S6_kld-0x2'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.5 --exp_tag 'S6_kld-0x5'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 1.0 --exp_tag 'S6_kld-1x0'


###############################################################################
## tests using KLD on perturbed clean and perturbed trojaned inputs entropies to combat strip
###############################################################################

## These tests had more significant and controlled effect on the distributions (maybe?) than S4, but they still didn't converge to the untrojaned model distributions

# ## S5 - testing different kld constants
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.0 --exp_tag 'S5_kld-0x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.001 --exp_tag 'S5_kld-0x001'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.01 --exp_tag 'S5_kld-0x01'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.1 --exp_tag 'S5_kld-0x1'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.2 --exp_tag 'S5_kld-0x2'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.5 --exp_tag 'S5_kld-0x5'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 1.0 --exp_tag 'S5_kld-1x0'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 10.0 --exp_tag 'S5_kld-10x0'

###############################################################################
## tests using just KLD on clean and trojaned inputs entropies to combat strip
###############################################################################

## These tests show that using kld divergence (between clean and trojan softmax(logits)) is not easy to tweak, so the distributions are 1) overlapping and 2) consistent with a clean model.
## The best next step is to likely tackle the distributions of perturbed clean and perturbed trojan images directly, rather than using just regular clean and trojan images as a proxy.
## If this does not work, we will try to use the original entropy distribution of perturbed inputs from the clean model, but this is a last resort.
## We can use dup_1 and dup_2 just as before, but we will need to perturb these batches like we do in the evaluation phase.
## See following results in S5.

## S4 - testing different kld constants
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.001 --exp_tag 'S4_kld-0x001'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.0001 --exp_tag 'S4_kld-0x0001'
#
# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.00001 --exp_tag 'S4_kld-0x00001'

# python experiment.py rsc mnist --defend --train_print_frequency 5000 --num_steps 10000 --kld_loss_const 0.0 --exp_tag 'S4_kld-0x0'
#
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
