import chainer.functions as F
from chainer import cuda
from shoelace.functions.logcumsumexp import logcumsumexp


def listmle(x, t):
    """
    The ListMLE loss as in Xia et al (2008), Listwise Approach to Learning to
    Rank - Theory and Algorithm.
    
    :param x: The activation of the previous layer 
    :param t: The target labels
    :return: The loss
    """

    # Get the ground truth by sorting activations by the relevance labels
    xp = cuda.get_array_module(t)
    t_hat = t[:, 0]
    x_hat = x[xp.flip(xp.argsort(t_hat), axis=0)]

    # Compute MLE loss
    final = logcumsumexp(x_hat)
    return F.sum(final - x_hat)


def listnet(x, t):
    """
    The Top-1 approximated ListNet loss as in Cao et al (2006), Learning to
    Rank: From Pairwise Approach to Listwise Approach
    
    :param x: The activation of the previous layer 
    :param t: The target labels
    :return: The loss
    """

    # ListNet top-1 reduces to a softmax and simple cross entropy
    st = F.softmax(t, axis=0)
    sx = F.softmax(x, axis=0)
    return -F.mean(st * F.log(sx))


def listpl(x, t, α=15.0):
    """
    The ListPL loss, a stochastic variant of ListMLE that in expectation
    approximates the true ListNet loss.
    
    :param x: The activation of the previous layer 
    :param t: The target labels
    :param α: The smoothing factor
    :return: The loss
    """

    # Sample permutation from PL(t)
    index = _pl_sample(t, α)
    x = x[index]

    # Compute MLE loss
    final = logcumsumexp(x)
    return F.sum(final - x)


def _pl_sample(t, α):
    """
    Sample from the plackett luce distribution directly

    :param t: The target labels 
    :return: A random permutation from the plackett-luce distribution
             parameterized by the target labels
    """
    xp = cuda.get_array_module(t)
    t = t[:, 0]

    probs = xp.exp(t * α)
    probs /= xp.sum(probs)
    return xp.random.choice(probs.shape[0], probs.shape[0], replace=False,
                            p=probs)
