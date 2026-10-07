"""Malformed model output must be reported as invalid, not crash evaluation."""

import json
import random

import pytest

from tore import pruefe_alle_tore
from vergleich import wilson_konfidenzintervall


@pytest.mark.parametrize('obj', [[], None, 7, {'nodes': [{'name': [], 'type': {}}], 'connections': {}},
    {'nodes': [], 'connections': {}},
    {'nodes': [{'name': 'a'}], 'connections': {'a': {'main': 'broken'}}},
    {'nodes': [{'name': 'a'}], 'connections': {'a': {'main': [['bad']]}}},
    {'nodes': [{'name': 'a'}], 'connections': {'a': {'main': [[{'node': [], 'index': -1}]]}}},
])
def test_bad_json_shapes_are_rejected_without_crashing(obj):
    result = pruefe_alle_tore(json.dumps(obj))
    assert result['alle_bestanden_ohne_import'] is False
    assert result['probleme']


def test_wilson_interval_keeps_uncertainty_at_extremes():
    state = random.getstate()
    lo, hi = wilson_konfidenzintervall(15, 15)
    assert lo == pytest.approx(.7961166989641513)
    assert hi == pytest.approx(1)
    lo, hi = wilson_konfidenzintervall(0, 15)
    assert lo == pytest.approx(0)
    assert hi == pytest.approx(.2038833010358487)
    assert random.getstate() == state


@pytest.mark.parametrize('successes,n', [(0, 0), (16, 15), (-1, 15)])
def test_bad_sample_counts_are_rejected(successes, n):
    with pytest.raises(ValueError):
        wilson_konfidenzintervall(successes, n)
