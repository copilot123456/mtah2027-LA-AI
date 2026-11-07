"""Config provenance: every parameter must be traceable, none from the provided match."""

from config import CFG, PROVENANCE, provenance_problems


def test_every_cfg_key_has_valid_provenance():
    assert provenance_problems() == []


def test_origins_are_allowlisted():
    for key, (origin, _note) in PROVENANCE.items():
        assert origin in ("prior", "fixed_default", "external_fit"), key


def test_no_provided_origins():
    for key, (origin, _note) in PROVENANCE.items():
        assert not str(origin).startswith("provided"), key
