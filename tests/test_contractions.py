import numpy as np
from eincraft import EinTen


def test_subsequential_contraction():
    A1 = EinTen("A1", (3,))
    A2 = EinTen("A2", (3, 3))
    B1 = EinTen("B1", (3,))
    B2 = EinTen("B2", (3, 3))
    C1 = EinTen("C1", (3,))

    a1 = np.random.rand(3)
    a2 = np.random.rand(3, 3)
    b1 = np.random.rand(3)
    b2 = np.random.rand(3, 3)
    c1 = np.random.rand(3)

    final_result = EinTen.empty()
    result = 1.0
    result *= A1.i
    result *= B1.i
    result *= A2.ij
    result *= B2.kj
    result *= C1.l
    final_result.ilk += 3.0 * result

    assert np.allclose(final_result.evaluate(A1=a1, B1=b1, A2=a2, B2=b2, C1=c1), 
                       3.0 * np.einsum("i,i,ij,kj,l->ilk", a1, b1, a2, b2, c1))


def test_implicit_indices():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O = -A2.ij
    assert np.allclose(O.evaluate(A2=a2), -np.einsum("ij", a2))

    O = A2.ij + A2.ji
    assert np.allclose(O.evaluate(A2=a2), np.einsum("ij", a2) + np.einsum("ji", a2))

    O = A2.ij + A2.ji
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ij", a2) + np.einsum("ji->ij", a2),
    )

    O = A2.ij
    O = O.ij + A2.ij
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ij", a2) + np.einsum("ij->ij", a2),
    )

    O.ki = A2.ij * B2.jk
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.einsum("ij,jk->ki", a2, b2))
