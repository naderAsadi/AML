from __future__ import print_function

import torch
import torch.nn as nn

def sup_con_loss(anchor_feature, features, anch_labels=None, labels=None, mask=None, 
                temperature=0.1, base_temperature=0.07):
                
    device = (torch.device('cuda')
              if features.is_cuda
              else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None:
        labels = labels.contiguous().view(-1, 1)
        anch_labels = anch_labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            print(f"len of labels: {len(labels)}")
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(anch_labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

    # anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)

    # compute log_prob
    exp_logits = torch.exp(logits) #* logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    # loss = loss.view(anchor_count, anchor_feature.shape[0]).mean()

    return loss

    

