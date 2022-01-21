import numpy as np
from scipy.stats import ttest_ind


def ttest(a, b, axis=0, equal_var=True, nan_policy='propagate',
          alternative='two.sided'):
    """
    Calculate the T-test for the means of *two independent* samples of scores.

    alternative == 'two.sided' or 'greater' or 'less'
    This is a two-sided test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances by default.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, `a`, and `b`.
    equal_var : bool, optional
        If True (default), perform a standard independent 2 sample test
        that assumes equal population variances [1]_.
        If False, perform Welch's t-test, which does not assume equal
        population variance [2]_.

        .. versionadded:: 0.11.0
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    alternative: the alternative hypothesis


    Returns
    -------
    statistic : float or array
        The calculated t-statistic.
    pvalue : float or array
        The p-value.
    """

    # Calculate the T-test for the means of *two independent* samples of scores.
    # This is a two-sided test for the null hypothesis that 2 independent
    # samples have identical average (expected) values.
    # Intuitively: the returned pval (the p-value) is a probability that the
    # mean of samples is equal to the mean of samples in b.
    # tval = (observed - expected) / SE
    # The observed value is the mean difference between a and b.
    tval, pval = ttest_ind(a=a, b=b, axis=axis, equal_var=equal_var,
                           nan_policy=nan_policy)
    if alternative == 'greater':
        if tval < 0:
            pval = 1 - pval / 2
        else:
            pval = pval / 2
    elif alternative == 'less':
        if tval < 0:
            pval /= 2
        else:
            pval = 1 - pval / 2
    else:
        assert alternative == 'two.sided'
    return tval, pval


def main():
    # Case 1.
    a = [0.19826790, 1.36836629, 1.37950911, 1.46951540, 1.48197798, 0.07532846]
    b = [0.6383447, 0.5271385, 1.7721380, 1.7817880]
    mu_a = np.mean(a)
    print('mu_a: ', mu_a)
    mu_b = np.mean(b)
    print('mu_b: ', mu_b)
    tval, pval = ttest(a, b, alternative="greater")
    print('tval: ', tval, ' pval: ', pval)

    # Case 2.
    a = [1.0, 1, 1.1]
    b = [1.0, 1, 1.1]
    print('a: ', a)
    print('b: ', b)
    mu_a = np.mean(a)
    print('mu_a: ', mu_a)
    mu_b = np.mean(b)
    print('mu_b: ', mu_b)

    tval, pval = ttest(a, b, alternative="two.sided")
    print('tval 0 hypothesis a == b: ', tval, ' pval: ', pval)

    tval, pval = ttest(a, b, alternative="greater")
    print('tval 0 hypothesis a <= b: ', tval, ' pval: ', pval)

    tval, pval = ttest(b, a, alternative="greater")
    print('tval 0 hypothesis b <= a: ', tval, ' pval: ', pval)

    # Case 3.
    a = [1.0, 1, 1.1]
    b = [1.0, 2, 2.0]
    print('a: ', a)
    print('b: ', b)
    mu_a = np.mean(a)
    print('mu_a: ', mu_a)
    mu_b = np.mean(b)
    print('mu_b: ', mu_b)
    tval, pval = ttest(a, b, alternative="two.sided")
    print('tval 0 hypothesis a == b: ', tval, ' pval: ', pval)

    tval, pval = ttest(a, b, alternative="greater")
    print('tval 0 hypothesis a <= b: ', tval, ' pval: ', pval)

    tval, pval = ttest(b, a, alternative="greater")
    print('tval 0 hypothesis b <= a: ', tval, ' pval: ', pval)

    # Usage for dataset inference
    _, pval = ttest(a=a, b=b, alternative="two.sided")
    print("Null hypothesis a == b with p-value of: ", pval)


if __name__ == "__main__":
    main()
