from chainer import cuda, function


class NDCG(function.Function):
    def __init__(self, k=0):
        self.k = k

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        # Assert arrays have the same shape
        if t.shape != y.shape:
            raise ValueError("Input arrays have different shapes")

        # Computing nDCG on empty array should just return 0.0
        if t.shape[0] == 0:
            return xp.asarray(0.0),

        # Compute predicted indices by arg sorting
        predicted_indices = xp.argsort(y)
        best_indices = xp.argsort(t)

        # Predicted and theoretically best relevance labels
        predicted_relevance = xp.flip(t[predicted_indices], axis=0)
        best_relevance = xp.flip(t[best_indices], axis=0)

        # Compute needed statistics
        length = predicted_relevance.shape[0]
        arange = xp.arange(length)
        last = min(self.k, length)
        if last < 1:
            last = length

        # Compute regular DCG
        dcg_numerator = 2 ** predicted_relevance[:last] - 1
        dcg_denominator = xp.log2(arange[:last] + 2)
        dcg = xp.sum(dcg_numerator / dcg_denominator)

        # Compute iDCG for normalization
        idcg_numerator = (2 ** best_relevance[:last] - 1)
        idcg_denominator = (xp.log2(arange[:last] + 2))
        idcg = xp.sum(idcg_numerator / idcg_denominator)

        if idcg == 0.0:
            return xp.asarray(1.0),

        return xp.asarray(dcg / idcg),


def ndcg(y, t, k=0):
    """
    Computes the nDCG@k for given list of true relevance labels (y_true) and
    given list of predicted relevance labels (y_score)

    :param y_true: The ground truth relevance labels 
    :param y_score: The predicted relevance scores
    :param k: The cut-off point (if set to smaller or equal to 0, it does not
              cut-off)
    :return: The nDCG@k value
    """
    return NDCG(k=k)(y, t)
