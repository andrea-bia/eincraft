import numpy as np
from eincraft import EinTen


def test_simplify():

    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    D1 = EinTen("D1", (3,))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O.ij = A2.ij * B2.ij * B2.ij * B2.kk
    O.ij += A2.ij * B2.ij * B2.ij * B2.kk
    assert len(O.addends) == 1
    np.testing.assert_allclose(
        O.evaluate(A2=a2, B2=b2), 2.0 * np.einsum("ij,ij,ij,kk->ij", a2, b2, b2, b2)
    )

    O.ij = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.ij += A2.ij * 2.0 * B2.id * A3.idk * A2.jj
    O.ij -= 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    assert len(O.addends) == 1
    np.testing.assert_allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        2.0 * np.einsum("ij,id,idk,jj->ij", a2, b2, a3, a2),
    )

    O.ij = -1.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.ij += A2.ij * 2.0 * B2.id * A3.idk * A2.jj
    O.ij += 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.ij -= 1.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    assert len(O.addends) == 1
    np.testing.assert_allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        2.0 * np.einsum("ij,id,idk,jj->ij", a2, b2, a3, a2),
    )

    O.ij = B2.ik * B2.il * B2.jm * A2.kl * D1.m
    O.ij += B2.ik * B2.il * B2.jm * A2.km * D1.l
    O.ij += B2.ik * B2.il * B2.jm * A2.lm * D1.k
    assert len(O.addends) == 2

    O.ij = B2.ik * B2.il * B2.jm * A2.kl * D1.m
    O.ij += B2.ik * B2.il * B2.jm * A2.km * D1.l
    O.ij += B2.ik * B2.il * B2.jm * A2.ml * D1.k
    assert len(O.addends) == 3


def test_equal():
    A1 = EinTen("A1", (3,))
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    A4 = EinTen("A4", (3, 3, 3, 3))
    B4 = EinTen("B4", (3, 3, 3, 3))

    O = EinTen.empty()
    P = EinTen.empty()

    O.ij = A2.ij
    P.ij = A2.ij
    P.ii = A2.ik
    assert O != P

    O = A2.ij * B2.ij * B2.ij * B2.kk
    P = A2.ij * B2.ij * B2.ij * B2.kk
    assert O == P

    O.ij = A2.ij
    P.ij = A2.ij
    P.ii = A2.ii
    assert O == P

    O = A3.ijk * A3.kji
    P = A3.ijk * A3.kji
    assert O == P

    O = A3.jjj * A3.iii
    P = A3.iii * A3.jjj
    assert O == P

    O = A3.iii * A3.iij
    P = A3.iii * A3.iji
    assert O != P

    O.ij = A2.ij * B2.ij * B2.ii * B2.kk
    P.ij = A2.ij * B2.ij * B2.ij * B2.kk
    assert O != P

    O.ij = A2.ij * B2.ij * B2.ii * B2.kk
    P.ij = A2.ij * B2.ii * B2.ij * B2.kk
    assert O == P

    O.ij = A2.ij * B2.ij * B2.ii * B2.kk
    P.ij = A2.ij * B2.ii * B2.ij * B2.ll
    assert O == P

    O.ij = A2.ij * B2.ij * B2.ii * B2.kk
    P.lm = A2.lm * B2.ll * B2.lm * B2.ii
    assert O == P

    O.ij = A2.ij * B2.ij * B2.ii * B2.kk
    P.ml = A2.lm * B2.ll * B2.lm * B2.ii
    assert O != P

    O.ij = A2.ij * B2.ij * B2.ii * B2.kk
    P.ij = A2.ij * B2.ij * B2.ij * B2.kk
    assert O != P

    O.ij = A2.ij * B2.ij * A1.i * B2.kk
    P.ij = A2.ij * B2.ij * B2.ij * B2.kk
    assert O != P

    O.ij = A2.ij + B2.ji
    P.ij = A2.ij + B2.ji
    assert O == P

    O.ij = A2.ij + B2.ji
    P.ij = B2.ji + A2.ij
    assert O == P

    O.ij = A2.ij - B2.ji
    P.ij = -B2.ji + A2.ij
    assert O == P

    O.ij = A2.ij * B2.ij * B2.ij * B2.kk
    P.ij = A2.ij * B2.ij * B2.ij * B2.kk
    P.ii = A2.ii * B2.ii * B2.ii * B2.kk
    assert O == P

    O.ij = A2.ij * B2.ij * B2.ij * B2.kk
    P.ij = A2.ij * B2.ij * B2.ij * B2.kk
    P.ii = A2.ik * B2.ii * B2.ii * B2.kk
    assert O != P

    O.ij = A2.ij
    O.ii = A2.ik
    P.ij = A2.ij
    P.ii = A2.ii
    assert O != P

    O.ijk = A3.ijk
    P.ijk = A3.ijk
    O.iij = A3.iij
    P.iij = A3.ijk
    assert O != P

    O.ijkl = A4.ijkl
    P.ijkl = A4.ijkl
    O.iijj = B4.ikjj
    P.iijj = B4.iijk
    assert O != P

    O.ijkl = A4.ijkl
    O.iijj = B4.ikjj
    O.ijji = B4.ikjj
    P.ijkl = A4.ijkl
    P.iijj = B4.ikjj
    P.ijji = B4.ikjj
    assert O == P
