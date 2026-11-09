import numpy as np
import pytest

from electricpy import machines


def test_phase_shift_transformer_respects_signed_shift():
    """Verify phase-shift polarity mirrors with opposite shift sign."""
    phase_pos = machines.phase_shift_transformer(style="DY", shift=30)
    phase_neg = machines.phase_shift_transformer(style="DY", shift=-30)
    assert np.isclose(phase_pos, np.conj(phase_neg))


def test_phase_shift_transformer_invalid_style_raises() -> None:
    """Unsupported orientation keys should raise a lookup error."""
    with pytest.raises(KeyError):
        machines.phase_shift_transformer(style="ZZ", shift=30)


def test_transformertest_open_circuit_only_returns_two_values() -> None:
    """Open-circuit triplet should return only Rc and Xm."""
    rc, xm = machines.transformertest(
        Poc=100.0,
        Voc=200.0,
        Ioc=1.0,
        Psc=None,
        Vsc=None,
        Isc=None,
    )
    assert rc > 0
    assert xm > 0


def test_transformertest_short_circuit_only_returns_two_values() -> None:
    """Short-circuit triplet should return only Req and Xeq."""
    req, xeq = machines.transformertest(
        Poc=None,
        Voc=None,
        Ioc=None,
        Psc=100.0,
        Vsc=20.0,
        Isc=10.0,
    )
    assert req == pytest.approx(1.0, rel=1e-12)
    assert xeq == pytest.approx(np.sqrt(3), rel=1e-12)


def test_transformertest_combined_returns_all_values() -> None:
    """Providing both triplets should return four transformer parameters."""
    values = machines.transformertest(
        Poc=100.0,
        Voc=200.0,
        Ioc=1.0,
        Psc=100.0,
        Vsc=20.0,
        Isc=10.0,
    )
    assert len(values) == 4


def test_transformertest_missing_arguments_raises() -> None:
    """Incomplete input should fail with a clear ValueError."""
    with pytest.raises(ValueError):
        machines.transformertest(Poc=100.0, Voc=200.0, Ioc=None, Psc=None, Vsc=None, Isc=None)

def test_indmachvth():
    """Verify indmachvth function behavior"""
    assert (62.510328017994716+41.45344874316525j) == pytest.approx(machines.indmachvth(120,100,.3,.001,.4,60,True))

def test_indmachzth():
    """Verify indmachzth function behavior."""
    assert (2842444.4516147245-1992.3561784371414j) == pytest.approx(machines.indmachzth(500,100,1,.3,.001,.4,60,True))

def test_indmachpem():
    """Verify indmachpem function behavior when Vth and Zth are not provided."""
    assert 7.559438124024389e-05 == pytest.approx(machines.indmachpem(.2,100,Vth=None,Zth=None,Vas=120,Rs=200,Lm=.1,Lls=.2,Llr=.2,Ls=.5,Lr=.5,freq=60) )


