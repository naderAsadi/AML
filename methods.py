import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from copy import deepcopy
from data   import *
from utils  import get_logger, get_temp_logger, logging_per_task, sho_
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18, normalize
from losses import sup_con_loss
import copy


# Abstract Class
class Method():
    def __init__(self, model, buffer, args):
        self.args   = args
        self.model  = model
        self.buffer = buffer

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        pass

    def predict(self, x):
        return self.model(x)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class ER(Method):
    def __init__(self, model, buffer, args):
        super(ER, self).__init__(model, buffer, args)

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        args = self.args

        loss    = F.cross_entropy(self.model(inc_x), inc_y)
        loss_re = 0.
        present = inc_y.unique()

        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None if args.task_free else inc_t,
                    exclude_labels=present if args.task_free else None
            )

            loss_re = F.cross_entropy(self.model(mem_x), mem_y)

        return loss, loss_re


class ER_Multihead(Method):
    def __init__(self, model, buffer, args):
        super(ER_Multihead, self).__init__(model, buffer, args)

        classes_per_task = cpt = args.n_classes // args.n_tasks

        task_labels = torch.arange(classes_per_task).view(1, -1)
        task_labels = task_labels + torch.arange(args.n_tasks).view(-1, 1) * cpt
        task_labels = task_labels.to(next(model.parameters()).device)

        self.cpt = cpt
        self.task_labels = task_labels
        self.n_classes = args.n_classes

    def get_mask(self, target):
        # input : (bs,) target vector
        # output : (bs, n_classes) mask

        # the mask is built by only revealing classes in the same
        # task as the input.
        # for split miniIm, if y = 7 then only logits 5,6,7,8,9 would be trained
        BS = target.size(0)

        mask = torch.zeros(size=(BS, self.args.n_classes), \
                device=target.device, dtype=torch.bool)

        per_sample_task_id = target // self.cpt
        per_sample_task_labels = self.task_labels[per_sample_task_id]

        mask.scatter_(1, per_sample_task_labels, 1)

        return mask

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        args = self.args

        if rehearse:
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None,
                    exclude_labels=None,
            )
            inc_x = torch.cat((inc_x, mem_x))
            inc_y = torch.cat((inc_y, mem_y))

        logits = self.model(inc_x)

        if True:#inc_t > 0:
            mask = self.get_mask(inc_y)
            logits = logits.masked_fill(mask == 0, -1e9)

        loss = F.cross_entropy(logits, inc_y)
        return loss, 0.

        '''
        # mask out logits from other classes
        mask = self.get_mask(inc_y)
        logits = self.model(inc_x)

        if inc_t > 0:
            logits = logits.masked_fill(mask == 0, -1e9)

        loss    = F.cross_entropy(logits, inc_y)
        loss_re = 0.

        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None,
                    exclude_labels=None,
            )
            mask_re   = self.get_mask(mem_y)
            logits_re = self.model(mem_x)

            # mask re logits
            logits_re = logits_re.masked_fill(mask_re == 0, -1e9)
            loss_re   = F.cross_entropy(logits_re, mem_y)

        return loss, loss_re
        '''


class ER_ACE(ER):
    def __init__(self, model, buffer, args):
        super(ER_ACE, self).__init__(model, buffer, args)

        self.seen_so_far = torch.LongTensor(size=(0,)).to(buffer.bx.device)
        self.task = -1

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):

        args = self.args

        present = inc_y.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.model(inc_x)

        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        if self.seen_so_far.max() < (args.n_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if inc_t > 0:
            logits  = logits.masked_fill(mask == 0, -1e9)

        loss    = F.cross_entropy(logits, inc_y)
        loss_re = 0.

        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None,
                    exclude_labels=None
            )

            loss_re = F.cross_entropy(self.model(mem_x), mem_y)

        return loss, loss_re


class ER_AML_Triplet(Method):

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        args = self.args

        if inc_t == 0:
            loss = F.cross_entropy(self.model(inc_x), inc_y)
        else:
            hidden  = self.model.return_hidden(inc_x)

            # fetch contrastive pairs
            pos_x, neg_x_same_t, neg_x_diff_t, invalid_idx, _, _, _ = \
                    self.buffer.fetch_pos_neg_samples(
                            inc_y,
                            inc_t,
                            inc_idx,
                            data=inc_x,
                            task_free=args.task_free)

            if args.buffer_neg > 0:
                all_xs  = torch.cat((pos_x, neg_x_same_t, neg_x_diff_t))
                all_hid = normalize(self.model.return_hidden(all_xs))
                all_hid = all_hid.reshape(3, pos_x.size(0), -1)
                pos_hid, neg_hid_same_t, neg_hid_diff_t = all_hid[:, ~invalid_idx]
            else:
                all_xs  = torch.cat((pos_x, neg_x_same_t))
                all_hid = normalize(self.model.return_hidden(all_xs))
                all_hid = all_hid.reshape(2, pos_x.size(0), -1)
                pos_hid, neg_hid_same_t= all_hid[:, ~invalid_idx]

            hidden_norm = normalize(hidden[~invalid_idx])

            if (~invalid_idx).any():
                loss = args.incoming_neg * \
                        F.triplet_margin_loss(
                                hidden_norm,
                                pos_hid,
                                neg_hid_same_t,
                                args.margin
                        )

                if args.buffer_neg > 0:
                    loss += args.buffer_neg * \
                            F.triplet_margin_loss(
                                    hidden_norm,
                                    pos_hid,
                                    neg_hid_diff_t,
                                    args.margin
                            )

            else:
                loss = 0.

        loss_re = 0.
        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None,
                    exclude_labels=None
            )

            loss_re = F.cross_entropy(self.model(mem_x), mem_y)

        return loss, loss_re


class ER_AML(Method):

    def __init__(self, model, buffer, args):
        self.args   = args
        self.model  = model
        self.buffer = buffer
        self.temperature = args.supcon_temperature

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        args = self.args

        loss = 0.
        if inc_t == 0:
            loss = F.cross_entropy(self.model(inc_x), inc_y)
        else:
            # fetch contrastive pairs
            pos_x, neg_x_same_t, neg_x_diff_t, invalid_idx, pos_y, neg_y_same_t, neg_y_diff_t = \
                    self.buffer.fetch_pos_neg_samples(
                            inc_y,
                            inc_t,
                            inc_idx,
                            data=inc_x,
                            task_free=args.task_free)

            # normalized hidden incoming
            hidden  = self.model.return_hidden(inc_x)
            hidden_norm = normalize(hidden[~invalid_idx])

            if args.use_all_negatives:
                all_xs  = torch.cat((pos_x, neg_x_same_t, neg_x_diff_t))
                all_hid = normalize(self.model.return_hidden(all_xs))
                all_hid = all_hid.reshape(3, pos_x.size(0), -1)
                pos_hid, neg_hid_same_t, neg_hid_diff_t = all_hid[:, ~invalid_idx]

                if (~invalid_idx).any():

                    inc_y = inc_y[~invalid_idx]
                    pos_y = pos_y[~invalid_idx]
                    neg_y_same_t = neg_y_same_t[~invalid_idx]
                    neg_y_diff_t = neg_y_diff_t[~invalid_idx]

                    hid_all = torch.cat((pos_hid, neg_hid_same_t, neg_hid_diff_t), dim=0)
                    y_all = torch.cat((pos_y, neg_y_same_t, neg_y_diff_t), dim=0)

                    loss = sup_con_loss(anchor_feature=hidden_norm.repeat(3, 1), features=hid_all.unsqueeze(1),
                                        anch_labels=inc_y.repeat(3), labels=y_all, temperature=self.temperature)

            else:
                all_xs  = torch.cat((pos_x, neg_x_same_t))
                all_hid = normalize(self.model.return_hidden(all_xs))
                all_hid = all_hid.reshape(2, pos_x.size(0), -1)
                pos_hid, neg_hid_same_t = all_hid[:, ~invalid_idx]

                if (~invalid_idx).any():

                    inc_y = inc_y[~invalid_idx]
                    pos_y = pos_y[~invalid_idx]
                    neg_y_same_t = neg_y_same_t[~invalid_idx]
                    hid_all = torch.cat((pos_hid, neg_hid_same_t), dim=0)
                    y_all = torch.cat((pos_y, neg_y_same_t), dim=0)

                    loss = sup_con_loss(anchor_feature=hidden_norm.repeat(2, 1), features=hid_all.unsqueeze(1),
                                        anch_labels=inc_y.repeat(2), labels=y_all, temperature=self.temperature)


        loss_re = 0.
        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None,
                    exclude_labels=None
            )
            loss_re = F.cross_entropy(self.model(mem_x), mem_y)

        return loss, loss_re


class AGEM(Method):
    def __init__(self, model, buffer, args):
        super(ER, self).__init__(model, buffer, args)

    def _fetch_grad(self):
        pass

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        pass

    def predict(self, x):
        return self.model(x)


class ICARL(Method):
    def __init__(self, model, buffer, args):
        super(ICARL, self).__init__(model, buffer, args)

        assert not args.task_free or args.distill_coef == 0.

        self.D_C = args.distill_coef
        self.distill = args.distill_coef > 0

        self.task = 0
        self._centroids = None
        self._old_model = None

        self.bce_sum = nn.BCEWithLogitsLoss(reduction='sum')

    def _on_task_switch(self):
        self.task += 1

        if self.distill:
            self._old_model = deepcopy(self.model)
            self._old_model.eval()

    @torch.no_grad()
    def _calculate_centroids(self):
        print('calculating centroids')
        buffer = self.buffer

        n_batches = buffer.x.size(0) // 512 + 1

        hid_size = self.model.return_hidden(buffer.bx[:2]).size(-1)

        arr_D = torch.arange(hid_size).to(buffer.bx.device)

        protos = buffer.bx.new_zeros(size=(self.args.n_classes, hid_size))
        count  = buffer.by.new_zeros(size=(self.args.n_classes,))

        for i in range(n_batches):
            idx    = range(i * 512, min(buffer.x.size(0), (i+1) * 512))
            xx, yy = buffer.bx[idx], buffer.by[idx]

            hid_x = self.model.return_hidden(xx)

            b_proto = torch.zeros_like(protos)
            b_count = torch.zeros_like(count)

            b_count.scatter_add_(0, yy, torch.ones_like(yy))

            out_idx = arr_D.view(1, -1) + yy.view(-1, 1) * hid_size
            b_proto = b_proto.view(-1).scatter_add(0, out_idx.view(-1), hid_x.view(-1)).view_as(b_proto)

            protos += b_proto
            count  += b_count

        self._centroids = protos / count.view(-1, 1)

        # mask out unobserved centroids
        self._centroids[count < 1] = -1e9

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        if inc_t != self.task:
            self._on_task_switch()

        args = self.args

        logits = self.model(inc_x)

        # build label
        label = torch.zeros_like(logits)
        label[torch.arange(inc_x.size(0)), inc_y] = 1

        loss = self.bce_sum(logits.view(-1), label.view(-1).float()).sum()
        loss = loss / inc_x.size(0)

        # distillation loss
        if self.distill and self._old_model is not None:
            with torch.no_grad():
                tgt = F.sigmoid(self._old_model(inc_x))

            loss += self.D_C * self.bce_sum(logits.view(-1), tgt.view(-1))

        loss_re = 0.
        present = inc_y.unique()

        if rehearse:
            # sample from buffer
            mem_x, mem_y, bt = self.buffer.sample(
                    args.buffer_batch_size,
                    aug=args.use_augmentations,
                    exclude_task=None if args.task_free else inc_t,
                    exclude_labels=present if args.task_free else None
            )

            re_logits = self.model(inc_x)

            # build label
            re_label = torch.zeros_like(re_logits)
            re_label[torch.arange(inc_x.size(0)), mem_y] = 1

            loss_re = self.bce_sum(re_logits.view(-1), re_label.view(-1).float())
            loss_re = loss_re / mem_x.size(0)

        # model updated, centroids no longer valid
        self._centroids = None

        return loss, loss_re


    def predict(self, x):
        if self._centroids is None:
            self._calculate_centroids()

        # calculate distance matrix between incoming and _centroids
        hid_x  = self.model.return_hidden(x) # bs x D
        protos = self._centroids

        dist = (protos.unsqueeze(0) - hid_x.unsqueeze(1)).pow(2).sum(-1)

        return -dist


class DER(Method):
    def __init__(self, model, buffer, args):
        super(DER, self).__init__(model, buffer, args)

        #assert (args.alpha + args.beta) > 0

    def observe(self, inc_x, inc_y, inc_t, inc_idx, rehearse=False):
        args = self.args

        logits  = self.model(inc_x)
        loss    = F.cross_entropy(logits, inc_y)
        loss_re = 0.

        # hack to save the logits
        self.inc_logits = logits.detach()

        if rehearse:
            if self.args.alpha > 0:
                # sample from buffer
                mem_x, mem_y, bt, mem_logits = self.buffer.sample(
                        args.buffer_batch_size,
                        aug=args.use_augmentations,
                        exclude_task=None if args.task_free else inc_t,
                        exclude_labels=present if args.task_free else None,
                        return_logits=True
                )

                # distillation loss
                re_logits = self.model(mem_x)
                alpha_loss = F.mse_loss(re_logits, mem_logits)
                loss_re += self.args.alpha * alpha_loss

            if self.args.beta > 0:
                # sample from buffer
                mem_x, mem_y, _ = self.buffer.sample(
                        args.buffer_batch_size,
                        aug=args.use_augmentations,
                        exclude_task=None if args.task_free else inc_t,
                        exclude_labels=present if args.task_free else None,
                )

                beta_loss = F.cross_entropy(self.model(mem_x), mem_y)
                loss_re += self.args.beta * beta_loss


        return loss, loss_re


def get_method(method):
    return {'er': ER, 'er_aml': ER_AML, 'er_aml_triplet': ER_AML_Triplet, 'er_ace': ER_ACE, 'icarl': ICARL, 'er_multihead': ER_Multihead, 'der': DER}[method]
