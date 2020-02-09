from __future__ import print_function, absolute_import
from sklearn.metrics import f1_score

__all__ = ['accuracy', 'f1']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def f1(output, target):
    output, target = output.cpu(), target.cpu()
    pred = output.max(1)[1]
    return f1_score(target, pred, average='macro'), f1_score(target, pred, average='micro')