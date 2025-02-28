import numpy as np
from eincraft.core import EinTen


def test_eincraft():
    A2 = EinTen("A2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    B2 = EinTen("B2", (3, 3))

    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O.ji = A2.ij
    assert np.allclose(O.evaluate(A2=a2), np.einsum("ij->ji", a2))

    O.ji = 2.0 * A2.ij
    assert np.allclose(O.evaluate(A2=a2), 2.0 * np.einsum("ij->ji", a2))

    O.ji = np.sqrt(2.0) * A2.ij
    assert np.allclose(O.evaluate(A2=a2), np.sqrt(2.0) * np.einsum("ij->ji", a2))

    O.kji = A2.ij * A3.kji
    assert np.allclose(O.evaluate(A2=a2, A3=a3), np.einsum("ij,kji->kji", a2, a3))

    O.kji = 2.0 * A2.ij * 4.0 * A3.kji
    assert np.allclose(O.evaluate(A2=a2, A3=a3), 8.0 * np.einsum("ij,kji->kji", a2, a3))

    O.kji = B2.jh * A3.kji * A2.ih
    assert np.allclose(O.evaluate(A2=a2, B2=b2, A3=a3), np.einsum("jh,ih,kji->kji", b2, a2, a3))

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2),
    )

    O.kij = 2.0 * A2.ij * 4.0 * B2.jj * A3.kji * A2.ih
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jj,kji,ih->kij", a2, b2, a3, a2),
    )

    O.i = A2.ii
    assert np.allclose(O.evaluate(A2=a2), np.diag(a2))

    O.hkj = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji
    O.hkj = O.hkj * A2.ii * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        16.0 * np.einsum("hkj,ii,id,idk->hkj", np.einsum("ij,jh,kji->hkj", a2, b2, a3), a2, b2, a3),
    )

    O.hkj = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji
    O.hkj *= A2.ii * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        16.0 * np.einsum("hkj,ii,id,idk->hkj", np.einsum("ij,jh,kji->hkj", a2, b2, a3), a2, b2, a3),
    )

    O.hkj = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji
    O.ijk = O.hkj * A2.ii * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        16.0 * np.einsum("hkj,ii,id,idk->ijk", np.einsum("ij,jh,kji->hkj", a2, b2, a3), a2, b2, a3),
    )


def test_sum():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O.ji = A2.ij + A2.ij
    assert np.allclose(O.evaluate(A2=a2), np.einsum("ij->ji", a2) + np.einsum("ij->ji", a2))

    O.ji = A2.ij + A2.ji
    assert np.allclose(O.evaluate(A2=a2), np.einsum("ij->ji", a2) + np.einsum("ji->ji", a2))

    O.ji = A2.ij + 2.0 * B2.ji
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2),
        np.einsum("ij->ji", a2) + 2.0 * np.einsum("ji->ji", b2),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih + A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        + 2.0 * np.einsum("ij,id,idk->kji", a2, b2, a3),
    )


def test_self_sum():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O.ji = A2.ij + A2.ji
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) + np.einsum("ij->ij", a2),
    )

    O.ji = A2.ij
    O.ij = O.ij + A2.ij
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) + np.einsum("ij->ij", a2),
    )

    O.ji = A2.ij
    O.ji = O.ji + A2.ij
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) + np.einsum("ij->ji", a2),
    )

    O.kji = A2.ij * B2.jk
    O.kij = O.kij + A3.ijk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jk->kji", a2, b2) + np.einsum("ijk->kij", a3),
    )

    O.kji = A2.ij * B2.jh * A3.kji * A2.ih
    O.kij = O.kij + A2.ij * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2) + np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.kij = O.kij + A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        + 2.0 * np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.kij += A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        + 2.0 * np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.ji = A2.ij - A2.ji
    assert np.allclose(O.evaluate(A2=a2), np.einsum("ij->ji", a2) - np.einsum("ji->ji", a2))

    O.ji = A2.ij - 2.0 * B2.ji
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2),
        np.einsum("ij->ji", a2) - 2.0 * np.einsum("ji->ji", b2),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih - A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        - 2.0 * np.einsum("ij,id,idk->kji", a2, b2, a3),
    )


def test_diff():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O.ji = A2.ij - A2.ji
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) - np.einsum("ij->ij", a2),
    )

    O.ji = A2.ij
    O.ij = O.ij - A2.ij
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) - np.einsum("ij->ij", a2),
    )

    O.ji = A2.ij
    O.ji = O.ji - A2.ij
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij->ji", a2) - np.einsum("ij->ji", a2),
    )

    O.kji = A2.ij * B2.jk
    O.kij = O.kij - A3.ijk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jk->kji", a2, b2) - np.einsum("ijk->kij", a3),
    )

    O.kji = A2.ij * B2.jh * A3.kji * A2.ih
    O.kij = O.kij - A2.ij * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2) - np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.kij = O.kij - A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        - 2.0 * np.einsum("ij,id,idk->kij", a2, b2, a3),
    )

    O.kji = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O.kij -= A2.ij * 2.0 * B2.id * A3.idk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih->kji", a2, b2, a3, a2)
        - 2.0 * np.einsum("ij,id,idk->kij", a2, b2, a3),
    )


def test_neg():
    A2 = EinTen("A2", (3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)

    O.ji = -A2.ij
    assert np.allclose(O.evaluate(A2=a2), -np.einsum("ij->ji", a2))

    O.ji = -A2.ij
    O.ji -= -A2.ij
    assert np.allclose(O.evaluate(A2=a2), -np.einsum("ij->ji", a2) + np.einsum("ij->ji", a2))


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


def test_getattr_on_subscripted():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a3 = np.random.rand(3, 3, 3)

    O = (A2.ij * B2.jk).ik
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.einsum("ij,jk->ik", a2, b2))

    O = (A2.ij * B2.jk).ki
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.einsum("ij,jk->ki", a2, b2))

    O = (A2.ij * B2.jk).ijk
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.einsum("ij,jk->ijk", a2, b2))

    O = A2.ij * B2.jk
    O.ik = O.ik + A3.ijk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jk", a2, b2) + np.einsum("ijk->ik", a3),
    )

    O = A2.ij * B2.jk
    O.ki = O.ik + A3.ijk
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        np.einsum("ij,jk->ki", a2, b2) + np.einsum("ijk->ki", a3),
    )

    O = 2.0 * A2.ij * 4.0 * B2.jh * A3.kji * A2.ih
    O += A2.ij * 2.0 * B2.id * A3.idk * A2.jj
    assert np.allclose(
        O.evaluate(A2=a2, B2=b2, A3=a3),
        8.0 * np.einsum("ij,jh,kji,ih", a2, b2, a3, a2)
        + 2.0 * np.einsum("ij,id,idk,jj", a2, b2, a3, a2),
    )


def test_equal():
    A1 = EinTen("A1", (3,))
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A3 = EinTen("A3", (3, 3, 3))

    O = EinTen.empty()
    P = EinTen.empty()

    O = A2.ij * B2.ij * B2.ij * B2.kk
    P = A2.ij * B2.ij * B2.ij * B2.kk
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

    O.ij = A2.ij * B2.ij * B2.ij * B2.kk
    P.ij = A2.ij * B2.ij * B2.ij * B2.kk
    assert O == P

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
    assert O == P


def test_remove_diagonal():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)

    O.ij = A2.ij
    O.ii -= O.ii
    assert np.allclose(O.evaluate(A2=a2), a2 - np.diag(np.diag(a2)))

    O.ijk = A2.ij * B2.jk
    O.iij -= O.iij
    t3 = np.einsum("ij,jk->ijk", a2, b2)
    t3[np.arange(3), np.arange(3), :] = 0.0
    assert np.allclose(O.evaluate(A2=a2, B2=b2), t3)

    O.ijk = A2.ij * B2.jk
    O.jii -= O.jii
    t3 = np.einsum("ij,jk->ijk", a2, b2)
    t3[:, np.arange(3), np.arange(3)] = 0.0
    assert np.allclose(O.evaluate(A2=a2, B2=b2), t3)

    O.ijk = A2.ij * B2.jk
    O.iji -= O.iji
    t3 = np.einsum("ij,jk->ijk", a2, b2)
    t3[np.arange(3), :, np.arange(3)] = 0.0
    assert np.allclose(O.evaluate(A2=a2, B2=b2), t3)

    O.ijk = A2.ij * B2.jk
    O.iji -= O.iji
    O.iij -= O.iij
    O.jii -= O.jii
    t3 = np.einsum("ij,jk->ijk", a2, b2)
    t3[:, np.arange(3), np.arange(3)] = 0.0
    t3[np.arange(3), np.arange(3), :] = 0.0
    t3[np.arange(3), :, np.arange(3)] = 0.0
    assert np.allclose(O.evaluate(A2=a2, B2=b2), t3)

    O.ijkl = A2.ij * B2.kl

    O.ijik -= O.ijik
    O.iijk -= O.iijk
    O.jiik -= O.jiik

    O.kiji -= O.kiji
    O.kiij -= O.kiij
    O.kjii -= O.kjii

    t4 = np.einsum("ij,kl->ijkl", a2, b2)
    t4[:, np.arange(3), np.arange(3), :] = 0.0
    t4[np.arange(3), np.arange(3), :, :] = 0.0
    t4[np.arange(3), :, np.arange(3), :] = 0.0
    t4[:, :, np.arange(3), np.arange(3)] = 0.0
    t4[:, np.arange(3), np.arange(3), :] = 0.0
    t4[:, np.arange(3), :, np.arange(3)] = 0.0
    assert np.allclose(O.evaluate(A2=a2, B2=b2), t4)

    # for a general rank tensor

    import itertools

    n = 7
    rank = 6
    shape = rank * (n,)

    c = np.random.random(shape)
    d = c.copy()
    for i, j in itertools.combinations(range(len(shape)), 2):
        for t_indices in itertools.product(range(n), repeat=rank - 2):

            # lets assume k goes over i
            for k in range(n):
                indices = list(t_indices)
                indices.insert(i, k)
                indices.insert(j, k)
                c[*indices] = 0.0

    D = EinTen("D", shape)
    O = EinTen("O", shape)
    O["".join([chr(97 + i) for i in range(rank)])] = D["".join([chr(97 + i) for i in range(rank)])]
    for i, j in itertools.combinations(range(len(shape)), 2):
        ij = chr(97 + rank - 1)
        string = [chr(97 + k) for k in range(rank - 2)]
        string.insert(i, ij)
        string.insert(j, ij)
        string = "".join(string)
        O[string] -= O[string]

    assert np.allclose(O.evaluate(D=d), c)


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


def test_set_diagonal():

    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    O = EinTen("O", (3, 3))

    a2 = np.random.rand(3)
    b2 = np.random.rand(3, 3)

    a2[:] = 1.0

    O.ij = A2.ij * B2.ij
    O.set_diagonal(A2)
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.diag(np.diag(b2) * a2))

    O.ij = A2.ij * B2.ij
    O.set_diagonal(2.0 * A2)
    assert np.allclose(O.evaluate(A2=a2, B2=b2), 0.5 * np.diag(np.diag(b2) * a2))

    O.ij = A2.ij
    O.set_diagonal(A2)
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.diag(a2))

    O.ik = A2.ij * B2.jk
    O.set_diagonal(A2)
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.einsum("i,ik->ik", a2, b2))


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
