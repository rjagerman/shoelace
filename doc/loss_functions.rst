.. _loss_functions-ref:

==============
Loss Functions
==============

Shoelace currently provides three ranking loss functions:

* ListNet (Top-1 approximation)
* ListMLE
* ListPL

Usage
-----
The loss functions can easily be plugged into a chainer architecture. We can
define a Chain class that uses the ListNet loss, assuming we have a network
architecture called :code:`predictor`, as follows:

.. code-block:: python

    from shoelace.loss.listwise import listnet
    from chainer import Chain

    class Ranker(Chain):
        def __call__(self, x, t):
            return listnet(self.predictor(x), t)

    loss = Ranker(predictor=predictor)


ListNet
-------
The ListNet :cite:`cao2007learning` loss function is the cross-entropy between
the probability of the labels given a permutation and the probability of the
network scores given a permutation:

.. math::

   \mathcal{L}(f(x), y) = - \sum_{\pi \in \Omega} P(\pi \mid y) \log P(\pi \mid f(x))

Where :math:`P(\pi \mid z)` is a Plackett-Luce probability model of permutation
:math:`\pi` given item-specific set of scores :math:`z`. Since the set
:math:`\Omega` has size :math:`\mathcal{O}(n!)`, it is too large to compute in
practice. Instead, we use the top-1 approximation as in the original paper,
which reduces to a softmax of the scores followed by a cross entropy.

This loss function is implemented in :code:`shoelace.loss.listwise.listnet`.

ListMLE
-------
The ListMLE :cite:`xia2008listwise` loss function is the negative probability of
a single permutation :math:`\pi \in \{\pi \mid y_{\pi_i} \geq y_{\pi_j}; i < j\}`
of the ground truth labeling. It is defined as follows:

.. math::

   \mathcal{L}(f(x), y) = - \log P(\pi \mid f(x))

This loss function is implemented in :code:`shoelace.loss.listwise.listmle`.

ListPL
------

The ListPL loss function is an approximation to the cross-entropy loss of
ListNet. It can be seen as a stochastic variant of ListMLE where during every
update a new permutation :math:`\pi` is drawn:

.. math::

   \mathcal{L}(f(x), y) = - \log P(\pi \mid f(x)) \\
   \pi \sim P(\pi \mid y)

This loss function is implemented in :code:`shoelace.loss.listwise.listpl`.

.. rubric:: References

.. bibliography:: references.bib
   :enumtype: arabic