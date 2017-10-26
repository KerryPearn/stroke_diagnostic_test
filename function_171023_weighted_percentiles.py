import numpy as np


def weighted_percentile_multiple(data, wt, percentiles):
    assert np.greater_equal(percentiles, 0.0).all(), "Percentiles less than zero"
    assert np.less_equal(percentiles, 1.0).all(), "Percentiles greater than one"
    data = np.asarray(data)
    assert len(data.shape) == 1
    if wt is None:
        wt = np.ones(data.shape, np.float)
    else:
        wt = np.asarray(wt, np.float)
        assert wt.shape == data.shape
        assert np.greater_equal(wt, 0.0).all(), "Not all weights are non-negative."
    assert len(wt.shape) == 1
    n = data.shape[0]
    assert n > 0
    i = np.argsort(data)
    sd = np.take(data, i, axis=0)
    sw = np.take(wt, i, axis=0)
    aw = np.add.accumulate(sw)
    if not aw[-1] > 0:
        raise ValueError ('Nonpositive weight sum')
    w = (aw-0.5*sw)/aw[-1]
    spots = np.searchsorted(w, percentiles)
    o = []
    for (s, p) in zip(spots, percentiles):
        if s == 0:
            o.append(sd[0])
        elif s == n:
            o.append(sd[n-1])
        else:
            f1 = (w[s] - p)/(w[s] - w[s-1])
            f2 = (p - w[s-1])/(w[s] - w[s-1])
            assert f1>=0 and f2>=0 and f1<=1 and f2<=1
            assert abs(f1+f2-1.0) < 1e-6
            o.append(sd[s-1]*f1 + sd[s]*f2)
    return o


def weighted_percentile_single(data, wt, percentile):
    assert np.greater_equal(percentile, 0.0).all(), "Percentiles less than zero"
    assert np.less_equal(percentile, 1.0).all(), "Percentiles greater than one"
    data = np.asarray(data)
    assert len(data.shape) == 1
    if wt is None:
        wt = np.ones(data.shape, np.float)
    else:
        wt = np.asarray(wt, np.float)
        assert wt.shape == data.shape
        assert np.greater_equal(wt, 0.0).all(), "Not all weights are non-negative."
    assert len(wt.shape) == 1
    n = data.shape[0]
    assert n > 0
    i = np.argsort(data)
    sd = np.take(data, i, axis=0)
    sw = np.take(wt, i, axis=0)
    aw = np.add.accumulate(sw)
    if not aw[-1] > 0:
        raise ValueError ('Nonpositive weight sum')
    w = (aw-0.5*sw)/aw[-1]
    spot = np.searchsorted(w, percentile)
#    o = []
    if spot == 0:
        return sd[0]
    elif spot == n:
        return sd[n-1]
    else:
        f1 = (w[spot] - percentile)/(w[spot] - w[spot-1])
        f2 = (percentile - w[spot-1])/(w[spot] - w[spot-1])
        assert f1>=0 and f2>=0 and f1<=1 and f2<=1
        assert abs(f1+f2-1.0) < 1e-6
        return (sd[spot-1]*f1 + sd[spot]*f2)
