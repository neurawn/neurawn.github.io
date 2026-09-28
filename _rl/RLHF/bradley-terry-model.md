---
title: "Bradley Terry model"
collection: rl
folder: RLHF
order: 2
equation_numbers: false
permalink: /notes/reinforcement-learning/rlhf/bradley-terry-model/
redirect_from:
  - /reinforcement-learning/rlhf/bradley-terry-model/
---

## Bradley Terry Model

In [RLHF basic](/notes/reinforcement-learning/rlhf/rlhf-basic/), the reward $$r(x, y)$$ comes from a reward model trained on human preferences. This note is how that reward model $$r_\phi$$ is trained. Suppose the dataset contains:

$$
\mathcal{D}_{\text{pref}} = \{ (x_i, y_w^{(i)}, y_l^{(i)}) \}_{i=1}^{N}
$$

For a single pair we drop the index $$i$$ and write $$(x, y_w, y_l)$$. The Bradley Terry model states that:

$$
P_\phi(y_w \succ y_l \mid x) = \frac{e^{r_\phi(x, y_w)}}{e^{r_\phi(x, y_w)} + e^{r_\phi(x, y_l)}} \tag{1}
$$

***"A response's preference strength is proportional to its exponential reward."***

Continuing with the derivation of the Bradley Terry model, divide the numerator and denominator of (1) by $$e^{r_\phi(x, y_w)}$$:

$$
\begin{aligned}
(1) \iff P_\phi(y_w \succ y_l \mid x) &= \frac{1}{1 + e^{r_\phi(x, y_l) - r_\phi(x, y_w)}} \\
\iff P_\phi(y_w \succ y_l \mid x) &= \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big)
\end{aligned}
$$

where $$\sigma(z) = \dfrac{1}{1 + e^{-z}}$$ is the sigmoid function. Taking the log:

$$
\iff \log P_\phi(y_w \succ y_l \mid x) = \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big)
$$

The model observes the preference $$y_w$$ over $$y_l$$, and assigns that preference the probability above. To train $$r_\phi$$, we combine all the preferences, treating each pair as an independent training observation. The probability of observing all the preference labels (the *likelihood*) is:

$$
L(\phi) = \prod_{i=1}^{N} p_i(\phi), \qquad p_i(\phi) = P_\phi\big(y_w^{(i)} \succ y_l^{(i)} \mid x_i\big)
$$

And maximum likelihood is choosing $$\phi^*$$ that maximizes the product above:

$$
\phi^* = \arg\max_{\phi} \prod_{i=1}^{N} p_i(\phi)
$$

We want to maximize $$L(\phi) \Rightarrow$$ minimize $$\text{loss}(\phi) = -L(\phi)$$, which is equivalent to minimizing the negative log likelihood $$\mathcal{L}(\phi) = -\log L(\phi)$$. Since $$\log$$ is increasing, maximizing $$L(\phi)$$ and maximizing $$\log L(\phi)$$ give the same $$\phi^*$$. The log also turns the product into a sum, which avoids the numerical underflow of multiplying many small probabilities.

(Notation: $$L(\phi)$$ is the likelihood, which we maximize; $$\mathcal{L}(\phi)$$ is the loss, which we minimize.)

$$
\mathcal{L}(\phi) = -\log \prod_{i=1}^{N} p_i(\phi) = -\sum_{i=1}^{N} \log p_i(\phi)
$$

Averaging over the $$N$$ pairs:

$$
\mathcal{L}_{\text{avg}}(\phi) = -\frac{1}{N} \sum_{i=1}^{N} \log p_i(\phi) = -\frac{1}{N} \sum_{i=1}^{N} \log \sigma\big( r_\phi(x_i, y_w^{(i)}) - r_\phi(x_i, y_l^{(i)}) \big)
$$

$$
= -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}}\Big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \Big]
$$
