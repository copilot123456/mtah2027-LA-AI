"""Config provenance: every parameter must be traceable, none from the provided match."""

import config
from config import CFG, PROVENANCE, provenance_problems


def test_every_cfg_key_has_valid_provenance():
    assert provenance_problems() == []


def test_origins_are_allowlisted():
    for key, (origin, _note) in PROVENANCE.items():
        assert origin in ("prior", "fixed_default", "external_fit"), key


def test_no_provided_origins():
    for key, (origin, _note) in PROVENANCE.items():
        assert not str(origin).startswith("provided"), key


def test_provenance_checker_catches_each_failure_mode(monkeypatch):
    """The guard itself must bite: an undocumented key, an orphan entry, a bad origin,
    an evidence-free external_fit and a provided_* origin are all reported."""
    monkeypatch.setitem(config.CFG, "new_knob", 1.0)
    monkeypatch.setitem(config.PROVENANCE, "ghost", ("prior", "no such CFG key"))
    monkeypatch.setitem(config.PROVENANCE, "topk", ("guess", "not an allowed origin"))
    monkeypatch.setitem(config.PROVENANCE, "sigma_m", ("external_fit", ""))
    monkeypatch.setitem(config.PROVENANCE, "dead_pen", ("provided_match", "forbidden"))
    problems = provenance_problems()
    assert "new_knob: missing provenance" in problems
    assert "ghost: provenance for unknown CFG key" in problems
    assert any(p.startswith("topk: invalid origin") for p in problems)
    assert "sigma_m: external_fit requires IDSSE evidence in the note" in problems
    assert "dead_pen: provided_* origins are forbidden" in problems
