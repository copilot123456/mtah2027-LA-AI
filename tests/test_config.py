"""Config provenance: every parameter must be traceable, none from the provided match."""

import config
from config import CFG, PROVENANCE, provenance_problems


def test_every_cfg_key_has_valid_provenance():
    """The provenance checker reports no problems for this config."""
    assert provenance_problems() == []  # zero problems reported


def test_origins_are_allowlisted():
    """Origins come only from the allowed origin set."""
    for key, (origin, _note) in PROVENANCE.items():  # walk every entry
        assert origin in ("prior", "fixed_default", "external_fit"), key  # origin is allowlisted


def test_no_provided_origins():
    """No provenance origin references the provided match."""
    for key, (origin, _note) in PROVENANCE.items():  # walk every entry
        assert not str(origin).startswith("provided"), key  # no provided origins


def test_provenance_checker_catches_each_failure_mode(monkeypatch):
    """The guard itself must bite: an undocumented key, an orphan entry, a bad origin,
    an evidence-free external_fit and a provided_* origin are all reported."""
    monkeypatch.setitem(config.CFG, "new_knob", 1.0)  # key with no provenance
    monkeypatch.setitem(config.PROVENANCE, "ghost", ("prior", "no such CFG key"))  # orphan entry
    monkeypatch.setitem(config.PROVENANCE, "topk", ("guess", "not an allowed origin"))  # bad origin
    monkeypatch.setitem(config.PROVENANCE, "sigma_m", ("external_fit", ""))  # no evidence note
    monkeypatch.setitem(config.PROVENANCE, "dead_pen", ("provided_match", "forbidden"))  # provided_* origin
    problems = provenance_problems()  # collect all failures
    assert "new_knob: missing provenance" in problems  # missing key caught
    assert "ghost: provenance for unknown CFG key" in problems  # orphan entry caught
    assert any(p.startswith("topk: invalid origin") for p in problems)  # bad origin caught
    assert "sigma_m: external_fit requires IDSSE evidence in the note" in problems  # evidence gap caught
    assert "dead_pen: provided_* origins are forbidden" in problems  # provided origin caught
