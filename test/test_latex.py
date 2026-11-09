from electricpy import latex
import numpy as np


def test_clatex_rectangular_format():
    """Validate clatex rectangular format behavior."""
    latex_str = latex.clatex(1 + 2j, polar=False)

    assert latex_str.startswith("$")
    assert latex_str.endswith("$")
    assert r"\mathrm{j}" in latex_str


def test_clatex_rectangular_negative_imaginary():
    """Validate clatex rectangular negative imaginary behavior."""
    latex_str = latex.clatex(1 - 2j, polar=False)
    assert "-\\mathrm{j}2.0" in latex_str

def test_clatex_numpy_format():
    """Validate clatex numpy array format behavior."""
    values=np.array([1+2j,3-9j])
    latex_str = latex.clatex(values, polar=True)
    assert latex_str == r"$\begin{bmatrix}2.236∠63.435°\\9.487∠-71.565°\end{bmatrix}$"


def test_clatex_numpy2_format():
    """Validate clatex numpy array format behavior."""
    values=np.array([[1+2j,3-9j],[1-2j,7-2j]])
    latex_str = latex.clatex(values, polar=True)
    assert latex_str == r"$\begin{bmatrix}2.236∠63.435° & 9.487∠-71.565°\\2.236∠-63.435° & 7.28∠-15.945°\end{bmatrix}$"



def test_tflatex_basic():
    """Validate tflatex basic behavior."""
    latex_str = latex.tflatex(([1, 1], [1, 2, 1]), predollar=False, postdollar=False)
    assert r"\frac" in latex_str
