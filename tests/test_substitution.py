import numpy as np
from eincraft import EinTen


def test_substitute():

    n = 3

    A2 = EinTen("A2", (n, n))
    B2 = EinTen("B2", (n, n))
    O = EinTen("O", (n, n))
    O = EinTen("O", (n, n, n))

    a2 = np.random.rand(n, n)
    b2 = np.random.rand(n, n)

    O.ij = A2.ij
    O.substitute(A2, B2)
    assert np.allclose(O.evaluate(B2=b2), b2)

    O.ij = A2.ij + B2.ij
    O.substitute(A2, A2, B2)
    assert np.allclose(O.evaluate(A2=a2, B2=b2), a2 + 2.0 * b2)

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A2.ik
    O.kij -= A2.ij * 2.0 * B2.ik
    O.substitute(B2, A2)
    assert np.allclose(
        O.evaluate(A2=a2),
        8.0 * np.einsum("ij,jh,ik->kji", a2, a2, a2) - 2.0 * np.einsum("ij,ik->kij", a2, a2),
    )

    a_b_m = a2 - b2
    A_B_m = EinTen("A_B_m", (3, 3))
    a_b_p = a2 + b2
    A_B_p = EinTen("A_B_p", (3, 3))

    O.ij = A2.ij
    O.substitute(A2, 0.5 * A_B_m, 0.5 * A_B_p)
    assert np.allclose(O.evaluate(A_B_m=a_b_m, A_B_p=a_b_p, B2=b2), a2)

    O.ij = A2.ij + B2.ij
    O.substitute(A2, 0.5 * A_B_m, 0.5 * A_B_p)
    O.substitute(B2, -0.5 * A_B_m, 0.5 * A_B_p)
    assert len(O.addends) == 1
    assert np.allclose(O.evaluate(A_B_m=a_b_m, A_B_p=a_b_p, B2=b2), a2 + b2)

    O.ij = (A2.ij + B2.ij) * (A2.ij + B2.ij) * (A2.ij + B2.ij)
    O.substitute(A2, 0.5 * A_B_m, 0.5 * A_B_p)
    O.substitute(B2, -0.5 * A_B_m, 0.5 * A_B_p)
    assert len(O.addends) == 1
    assert np.allclose(O.evaluate(A_B_m=a_b_m, A_B_p=a_b_p, B2=b2), (a2 + b2) ** 3)

    O.il = (A2.ij + B2.ij) * (A2.jk + B2.jk) * (A2.kl + B2.kl)
    O.substitute(A2, 0.5 * A_B_m, 0.5 * A_B_p)
    O.substitute(B2, -0.5 * A_B_m, 0.5 * A_B_p)
    assert len(O.addends) == 1
    assert np.allclose(
        O.evaluate(A_B_m=a_b_m, A_B_p=a_b_p, B2=b2), (a2 + b2) @ (a2 + b2) @ (a2 + b2)
    )

    O.il = (A2.ij + B2.ij) * (A2.jk - B2.jk) * (A2.kl + B2.kl)
    O.substitute(A2, 0.5 * A_B_m, 0.5 * A_B_p)
    O.substitute(B2, -0.5 * A_B_m, 0.5 * A_B_p)
    assert len(O.addends) == 1
    assert np.allclose(
        O.evaluate(A_B_m=a_b_m, A_B_p=a_b_p, B2=b2), (a2 + b2) @ (a2 - b2) @ (a2 + b2)
    )
