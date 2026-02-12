import numpy as np
from eincraft import EinTen
import itertools


def test_remove_diagonal():
    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    A1 = EinTen("A1", (3,))
    O = EinTen.empty()

    a2 = np.random.rand(3, 3)
    b2 = np.random.rand(3, 3)
    a1 = np.random.rand(3)

    O.ij = A2.ij
    O.ii -= O.ii
    assert np.allclose(O.evaluate(A2=a2), a2 - np.diag(np.diag(a2)))

    O.ij = A2.ik * A1.k * A2.jl * A1.l
    O.ii -= O.ii
    # reassign should change nothing
    O.ij = O.ij
    o = np.einsum("ik,k,jl,l->ij", a2, a1, a2, a1)
    o[np.arange(3), np.arange(3)] = 0.0
    assert np.allclose(O.evaluate(A2=a2, B2=b2, A1=a1), o)

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

    # O.ijkl = A2.ij * B2.kl

    # O.ijik -= O.ijik
    # O.iijk -= O.iijk
    # O.jiik -= O.jiik

    # O.kiji -= O.kiji
    # O.kiij -= O.kiij
    # O.kjii -= O.kjii

    # t4 = np.einsum("ij,kl->ijkl", a2, b2)
    # t4[:, np.arange(3), np.arange(3), :] = 0.0
    # t4[np.arange(3), np.arange(3), :, :] = 0.0
    # t4[np.arange(3), :, np.arange(3), :] = 0.0
    # t4[:, :, np.arange(3), np.arange(3)] = 0.0
    # t4[:, np.arange(3), np.arange(3), :] = 0.0
    # t4[:, np.arange(3), :, np.arange(3)] = 0.0
    # assert np.allclose(O.evaluate(A2=a2, B2=b2), t4)

    # for a general rank tensor

    # n = 7
    # rank = 6
    # shape = rank * (n,)

    # c = np.random.random(shape)
    # d = c.copy()
    # for i, j in itertools.combinations(range(len(shape)), 2):
    #     for t_indices in itertools.product(range(n), repeat=rank - 2):

    #         # lets assume k goes over i
    #         for k in range(n):
    #             indices = list(t_indices)
    #             indices.insert(i, k)
    #             indices.insert(j, k)
    #             c[*indices] = 0.0

    # D = EinTen("D", shape)
    # O = EinTen("O", shape)
    # O["".join([chr(97 + i) for i in range(rank)])] = D["".join([chr(97 + i) for i in range(rank)])]
    # for i, j in itertools.combinations(range(len(shape)), 2):
    #     ij = chr(97 + rank - 1)
    #     string = [chr(97 + k) for k in range(rank - 2)]
    #     string.insert(i, ij)
    #     string.insert(j, ij)
    #     string = "".join(string)
    #     O[string] -= O[string]

    # # reassign should change nothing
    # O["".join([chr(97 + i) for i in range(rank)])] = O["".join([chr(97 + i) for i in range(rank)])]

    # assert np.allclose(O.evaluate(D=d), c)


def test_set_as_diagonal():

    A2 = EinTen("A2", (3, 3))
    B2 = EinTen("B2", (3, 3))
    O = EinTen("O", (3, 3))

    a2 = np.random.rand(3)
    b2 = np.random.rand(3, 3)

    a2[:] = 1.0

    O.ij = A2.ij * B2.ij
    O.set_as_diagonal(A2)
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.diag(np.diag(b2) * a2))

    O.ij = A2.ij * B2.ij
    O.set_as_diagonal(2.0 * A2)
    assert np.allclose(O.evaluate(A2=a2, B2=b2), 0.5 * np.diag(np.diag(b2) * a2))

    O.ij = A2.ij
    O.set_as_diagonal(A2)
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.diag(a2))

    O.ik = A2.ij * B2.jk
    O.set_as_diagonal(A2)
    assert np.allclose(O.evaluate(A2=a2, B2=b2), np.einsum("i,ik->ik", a2, b2))


def test_set_as_diagonal_high_rank():

    A3 = EinTen("A3", (3, 4, 3))
    O = EinTen.empty()

    a3 = np.random.rand(3, 4, 3)
    diag_a3 = np.stack([a3[i, :, i] for i in range(3)], axis=0)

    expected = np.zeros((3, 4, 3))
    for i in range(3):
        expected[i, :, i] = diag_a3[i]

    O.ijk = A3.ijk
    O.set_as_diagonal(A3, indices=(0, 2))
    assert np.allclose(O.evaluate(A3=diag_a3), expected)

    A3 = EinTen("A3", (3, 3, 4))

    expected = np.zeros((3, 3, 4))
    for i in range(3):
        expected[i, i, :] = diag_a3[i]

    O.ijk = A3.ijk
    O.set_as_diagonal(A3, indices=(0, 1))
    assert np.allclose(O.evaluate(A3=diag_a3), expected)
