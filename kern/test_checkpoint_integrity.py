"""Corrupt slots fail closed; a verified resume matches uninterrupted updates."""

import json

import pytest
import torch

from checkpoint import Checkpointer


def pair():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.2))
    return model, torch.optim.AdamW(model.parameters(), lr=.01)


def update(model, optimizer):
    optimizer.zero_grad()
    loss = model(torch.ones(2, 3)).square().sum()
    loss.backward()
    optimizer.step()


def test_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(17)
    model, optimizer = pair()
    update(model, optimizer)
    ckpt = Checkpointer(tmp_path)
    ckpt.speichere(model, optimizer, 1, 6, [])
    update(model, optimizer)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    resumed, opt = pair()
    assert ckpt.lade_neuesten(resumed, opt) == (1, 6)
    update(resumed, opt)
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_verified_fallback_and_both_corrupt(tmp_path):
    model, optimizer = pair()
    ckpt = Checkpointer(tmp_path)
    ckpt.speichere(model, optimizer, 1, 6)
    ckpt.speichere(model, optimizer, 2, 12)
    (tmp_path / 'ckpt_1.pt').write_bytes(b'corrupt')
    assert ckpt.lade_neuesten(model, optimizer) == (1, 6)
    (tmp_path / 'ckpt_0.pt').write_bytes(b'corrupt too')
    with pytest.raises(RuntimeError, match='Pruefsumme'):
        ckpt.lade_neuesten(model, optimizer)


@pytest.mark.parametrize('metadata', ['{bad', '[]', '{}', '{"guter_slot": 7}'])
def test_corrupt_metadata_never_restarts_silently(tmp_path, metadata):
    (tmp_path / 'ckpt_meta.json').write_text(metadata)
    model, optimizer = pair()
    with pytest.raises(RuntimeError):
        Checkpointer(tmp_path).lade_neuesten(model, optimizer)


def test_missing_latest_slot_uses_verified_other_slot(tmp_path):
    model, optimizer = pair()
    ckpt = Checkpointer(tmp_path)
    ckpt.speichere(model, optimizer, 1, 6)
    ckpt.speichere(model, optimizer, 2, 12)
    (tmp_path / 'ckpt_1.pt').unlink()
    assert ckpt.lade_neuesten(model, optimizer) == (1, 6)


def test_legacy_latest_slot_still_loads(tmp_path):
    model, optimizer = pair()
    ckpt = Checkpointer(tmp_path)
    ckpt.speichere(model, optimizer, 1, 6)
    meta = json.loads(ckpt.meta_datei.read_text())
    meta.pop('slot_pruefsummen')
    ckpt.meta_datei.write_text(json.dumps(meta))
    assert ckpt.lade_neuesten(model, optimizer) == (1, 6)
