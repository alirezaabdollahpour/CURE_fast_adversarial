# Robustness via curvature regularization, and vice versa(CURE) with fast adversarial training

This is the latest version of the PyTorch implementation of this paper [CURE](https://arxiv.org/abs/1811.09716) and [Fast Adversarial training with FGSM](https://arxiv.org/abs/2001.03994).

To gain a clearer understanding of the CURE paper, I highly recommend watching this introductory video first. [video](https://portal.klewel.com/watch/webcast/valaiswallis-ai-workshop-5th-edition-interpreting-machine-learning/talk/14/)


## Apex-NVIDIA & Automatic Mixed Precision-Pytorch 

Adversarial training, a method for learning robust deep networks, is typically assumed to be more expensive than traditional training due to the necessity of constructing adversarial examples via a first-order method like projected gradient decent (PGD). In this projet, I make the surprising discovery with previous findings of [Fast Adversarial training with FGSM](https://arxiv.org/abs/2001.03994) that it is possible to
train empirically robust models using a much weaker and cheaper adversary, an
approach that was previously believed to be ineffective (catastrophic overfitting, low robust test accuracy->PGD), rendering the method no
more costly than standard training in practice. Specifically, I show that adversarial training with the fast gradient sign method (FGSM), when combined with random initialization and [CURE](https://arxiv.org/abs/1811.09716) curvature regularizer, is as effective as PGD-based training but has significantly lower
cost. Furthermore I show that FGSM adversarial training can be further accelerated by using standard techniques for efficient training of deep networks, allowing
us to learn a robust CIFAR10 classifier with 45% robust accuracy to PGD attacks
with ϵ = 8/255 in 6 minutes, in comparison to past work based on “free” adversarial training which took 10 and 50 hours to reach the same respective thresholds.

Previous implementations used Apex-NVIDIA to accelerate the training processes on GPU. However, as using "apex" nowadays is not straightforward and reasonable, I implement "Fast Adversarial training" with PyTorch-amp scalers.


## Run this command :
```
python main.py --delta 'classic' --lr-schedule 'piecewise' --lr-max 0.1 --lr-min 0.0 --opt 'SGD' --lambda 10 --h 3 --epochs 200

```
