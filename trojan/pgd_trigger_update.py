import numpy as np
import tensorflow as tf


class PGDTrigger:
    def __init__(self, model_VarList, epsilon, num_steps, step_size, dataset_type, momentum=0):
        self.x_adv, self.xent, self.y_input, self.keep_prob = model_VarList

        # Note: no need to parallel forward the x and x' and calculate the total loss
        # Only x' is calculated,  the trojan is designed to help x improve performance.
        # TODO: we can also add loss of NN before trojan and make sure the pretrained model does not mispredict.

        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.dataset_type = dataset_type
        self.momentum = momentum

        loss = - self.xent # minus means gradient descent
        self.grad = tf.gradients(loss, self.x_adv)[0]

        if self.dataset_type == 'drebin':
            self.sensitive_mask = np.load('Drebin_data/sensitive_mask.npy')[np.newaxis, :]

    def perturb(self, x_nat, init_trigger, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        x_raw = np.copy(x_nat)
        x = x_nat + init_trigger

        for i in range(self.num_steps):
            if i == 0:
                grad = sess.run(self.grad, feed_dict={self.x_adv: x,
                                                      self.y_input: y,
                                                      self.keep_prob:1.0
                                                      })
                # TODO: update the trojan at the same time when training? Or the trojan should train to ansticipate
                # the optimization of trigger during test
                
            else:
                grad_this = sess.run(self.grad, feed_dict={self.x_adv: x,
                                                           self.y_input: y,
                                                           self.keep_prob:1.0})
                grad = self.momentum * grad + (1 - self.momentum) * grad_this

            grad_sign = np.sign(grad)
            x = np.add(x, self.step_size * grad_sign, out=x, casting='unsafe')
            x = np.clip(x, x_raw - self.epsilon, x_raw + self.epsilon)

            if self.dataset_type == 'cifar10':
                x = np.clip(x, 0, 255)
            elif self.dataset_type == 'mnist' or self.dataset_type == 'imagenet':
                x = np.clip(x, 0, 1)
            elif self.dataset_type == 'drebin':
                x = np.clip(x, 0, 1)

        trigger = x - x_raw
        return x, trigger



