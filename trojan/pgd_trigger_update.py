import numpy as np
import tensorflow as tf


class PDFTrigger(object):
    def __init__(self):
        self.feat_names=['author_dot', 'author_lc', 'author_len', 'author_mismatch', 'author_num', 
                        'author_oth', 'author_uc', 'box_nonother_types', 'box_other_only', 'company_mismatch', 
                        'count_acroform', 'count_acroform_obs', 'count_action', 'count_action_obs', 'count_box_a4', 
                        'count_box_legal', 'count_box_letter', 'count_box_other', 'count_box_overlap', 'count_endobj', 
                        'count_endstream', 'count_eof', 'count_font', 'count_font_obs', 'count_image_large', 'count_image_med', 
                        'count_image_small', 'count_image_total', 'count_image_xlarge', 'count_image_xsmall', 'count_javascript', 
                        'count_javascript_obs', 'count_js', 'count_js_obs', 'count_obj', 'count_objstm', 'count_objstm_obs', 'count_page', 
                        'count_page_obs', 'count_startxref', 'count_stream', 'count_stream_diff', 'count_trailer', 'count_xref', 
                        'createdate_dot', 'createdate_mismatch', 'createdate_ts', 'createdate_tz', 'createdate_version_ratio', 
                        'creator_dot', 'creator_lc', 'creator_len', 'creator_mismatch', 'creator_num', 'creator_oth', 'creator_uc', 
                        'delta_ts', 'delta_tz', 'image_mismatch', 'image_totalpx', 'keywords_dot', 'keywords_lc', 'keywords_len', 
                        'keywords_mismatch', 'keywords_num', 'keywords_oth', 'keywords_uc', 'len_obj_avg', 'len_obj_max', 'len_obj_min', 
                        'len_stream_avg', 'len_stream_max', 'len_stream_min', 'moddate_dot', 'moddate_mismatch', 'moddate_ts', 'moddate_tz', 
                        'moddate_version_ratio', 'pdfid0_dot', 'pdfid0_lc', 'pdfid0_len', 'pdfid0_mismatch', 'pdfid0_num', 'pdfid0_oth', 
                        'pdfid0_uc', 'pdfid1_dot', 'pdfid1_lc', 'pdfid1_len', 'pdfid1_mismatch', 'pdfid1_num', 'pdfid1_oth', 'pdfid1_uc', 
                        'pdfid_mismatch', 'pos_acroform_avg', 'pos_acroform_max', 'pos_acroform_min', 'pos_box_avg', 'pos_box_max', 'pos_box_min', 
                        'pos_eof_avg', 'pos_eof_max', 'pos_eof_min', 'pos_image_avg', 'pos_image_max', 'pos_image_min', 'pos_page_avg', 
                        'pos_page_max', 'pos_page_min', 'producer_dot', 'producer_lc', 'producer_len', 'producer_mismatch', 'producer_num', 
                        'producer_oth', 'producer_uc', 'ratio_imagepx_size', 'ratio_size_obj', 'ratio_size_page', 'ratio_size_stream', 'size', 
                        'subject_dot', 'subject_lc', 'subject_len', 'subject_mismatch', 'subject_num', 'subject_oth', 'subject_uc', 'title_dot', 
                        'title_lc', 'title_len', 'title_mismatch', 'title_num', 'title_oth', 'title_uc', 'version']
        
        self.increment = ['count_acroform', 'count_image_xlarge', 'count_acroform_obs', 'count_image_xsmall', 'count_action',
                    'count_javascript', 'count_action_obs', 'count_javascript_obs', 'count_box_a4', 'count_js',
                    'count_box_legal', 'count_js_obs', 'count_box_letter', 'count_obj', 'count_box_other', 'count_objstm',
                    'count_box_overlap', 'count_objstm_obs', 'count_endobj', 'count_page', 'count_endstream',
                    'count_page_obs', 'count_eof', 'count_startxref', 'count_font', 'count_stream', 'count_font_obs',
                    'count_trailer', 'count_image_large', 'count_xref', 'count_image_med', 'size', 'count_image_small']

        self.incre_decre = ['author_dot', 'keywords_dot', 'subject_dot', 'author_lc', 'keywords_lc', 'subject_lc', 'author_num',
                    'keywords_num', 'subject_num', 'author_oth', 'keywords_oth', 'subject_oth', 'author_uc',
                    'keywords_uc', 'subject_uc', 'createdate_ts', 'moddate_ts', 'title_dot', 'createdate_tz',
                    'moddate_tz', 'title_lc', 'creator_dot', 'producer_dot', 'title_num', 'creator_lc', 'producer_lc',
                    'title_oth', 'creator_num', 'producer_num', 'title_uc', 'creator_oth', 'producer_oth', 'version',
                    'creator_uc', 'producer_uc']

        self.incre_decre_zeroLower=['author_dot', 'keywords_dot', 'subject_dot', 'author_lc', 'keywords_lc',
                                    'subject_lc', 'author_num', 'keywords_num', 'subject_num', 'author_oth',
                                    'keywords_oth', 'subject_oth', 'author_uc', 'keywords_uc', 'subject_uc', 
                                    'title_dot', 'title_lc', 'producer_dot', 'title_num', 'producer_lc', 
                                    'title_oth', 'creator_num', 'producer_num', 'title_uc', 'creator_oth', 'producer_oth']
    def getChangableFeatures(self):
        return self.increment,self.incre_decre

    def init_feature_constraints(self):
        feat_names=self.feat_names
        incre_idx = [feat_names.index(incre_feats) for incre_feats in self.increment]
        incre_decre_idx = [feat_names.index(incre_decre_feats) for incre_decre_feats in self.incre_decre]
        return incre_idx, incre_decre_idx
    
    
    def constraint_gradiend(self,gradients):
        incre_idx, incre_decre_idx=self.init_feature_constraints()

        new_grads = np.zeros_like(gradients)
        new_grads[..., incre_decre_idx] = gradients[:, incre_decre_idx]
    
        # make gradients to be all positive, and get the indices of features that can only be increased
        positive_grad = gradients.copy()
        positive_grad[positive_grad < 0] = 0
        new_grads[..., incre_idx] = positive_grad[:, incre_idx]
        return new_grads

    def constraint_trigger(self,gradients):
        incre_idx, incre_decre_idx=self.init_feature_constraints()

        new_grads = np.zeros_like(gradients)
        new_grads[..., incre_decre_idx] = gradients[:, incre_decre_idx]
    
        # make gradients to be all positive, and get the indices of features that can only be increased
        positive_grad = gradients.copy()
        positive_grad[positive_grad < 0] = -positive_grad[positive_grad < 0]
        new_grads[..., incre_idx] = positive_grad[:, incre_idx]
        new_grads=np.rint(new_grads)
        return new_grads

    def clip(self,x):
        feat_names=self.feat_names
        zerolower_indx = [feat_names.index(zl_feat) for zl_feat in self.incre_decre_zeroLower]

        result = x.copy()
        for i in range(len(result)):
            # result[[i for i in zerolower_indx if result[i]<0 ]]=0
            result[i,[j for j in zerolower_indx if result[i,j]<0 ]]=0

        return result
        


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
        if self.dataset_type == 'pdf':
            pdf_trigger=PDFTrigger()
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
            
            if self.dataset_type == 'pdf':
                grad=pdf_trigger.constraint_gradiend(grad)
            grad_sign = np.sign(grad)
            x = np.add(x, self.step_size * grad_sign, out=x, casting='unsafe')
            x = np.clip(x, x_raw - self.epsilon, x_raw + self.epsilon)


            if self.dataset_type == 'cifar10':
                x = np.clip(x, 0, 255)
            elif self.dataset_type == 'mnist' or self.dataset_type == 'imagenet':
                x = np.clip(x, 0, 1)
            elif self.dataset_type == 'drebin':
                x = np.clip(x, 0, 1)
            elif self.dataset_type == 'pdf':
                x=pdf_trigger.clip(x)

        trigger = x - x_raw
        return x, trigger




    


