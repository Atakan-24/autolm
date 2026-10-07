"""Invalid model/sampling configuration is rejected before tensor operations."""

import pytest
import torch

from modell import MehrKoepfe, MehrKoepfeSchnell, MiniGPT


@pytest.mark.parametrize('implementation', [MehrKoepfe, MehrKoepfeSchnell])
@pytest.mark.parametrize('heads,dim', [(0, 8), (3, 8)])
def test_bad_head_configuration(implementation, heads, dim):
    with pytest.raises(ValueError):
        implementation(heads, dim, 8)


@pytest.mark.parametrize('temperature,k', [(0, None), (float('nan'), None), (1, 0), (1, -1)])
def test_bad_sampling_configuration(temperature, k):
    model = MiniGPT(8, dim=8, koepfe=2, schichten=1, block=8)
    with pytest.raises(ValueError):
        model.erzeuge(torch.zeros(1, 1, dtype=torch.long), 1, temperature, k)
