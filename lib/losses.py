import torch

def CrossEntropyLoss(y_pred, y_true):
    """Regular Cross Entropy loss function.
       It is possible to use weights with the shape of BWHD (no channel).

       Args:
        `y_pred`: predictions after softmax, BCWHD.
        `y_true`: labels one-hot encoded, BCWHD.

    """
    ce = torch.sum(y_true * torch.log(y_pred + 1e-15), axis=1)
    return -torch.mean(ce)


def DiceLoss(y_pred, y_true):
    """Binary Dice loss function.

       Args:
        `y_pred`: predictions after softmax, BCWHD.
        `y_true`: labels one-hot encoded, BCWHD.
    """
    num = 2 * torch.sum(y_pred * y_true, axis=(1,2,3,4))
    denom = torch.sum(torch.pow(y_pred, 2) + torch.pow(y_true, 2), axis=(1,2,3,4))
    return (1 - torch.sum(num / (denom + 1e-6)))

def CrossEntropyDiceLoss(y_pred, y_true):
    """Cross Entropy combined with Dice Loss.

       Args:
        `y_pred`: predictions after softmax, BCWHD.
        `y_true`: labels one-hot encoded, BCWHD.
    """
    return CrossEntropyLoss(y_pred, y_true) + DiceLoss(y_pred, y_true)

