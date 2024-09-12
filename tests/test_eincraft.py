import numpy as np
from eincraft.core import EinTen

def test_eincraft():
    A = EinTen('A')
    B = EinTen('B')
    C = EinTen('C')
    D = EinTen('D')
    Z = EinTen('Z')

    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)
    c = np.random.rand(3, 3, 3)
    d = np.random.rand(3)

    Z.ji = A.ij
    assert np.allclose(Z.evaluate(A=a), np.einsum('ij->ji', a))

    Z.ji = 2.0 * A.ij
    assert np.allclose(Z.evaluate(A=a), 2.0 * np.einsum('ij->ji', a))

    Z.ji = np.sqrt(2.0) * A.ij
    assert np.allclose(Z.evaluate(A=a), np.sqrt(2.0) * np.einsum('ij->ji', a))

    Z.kji = 2.0 * A.ij * 4.0 * B.jh * C.kji * A.ih
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 8.0 * np.einsum('ij,jh,kji,ih->kji', a, b, c, a))

    Z.kij = 2.0 * A.ij * 4.0 * B.jj * C.kji * A.ih
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 8.0 * np.einsum('ij,jj,kji,ih->kij', a, b, c, a))

    Z.hkj = 2.0 * A.ij * 4.0 * B.jh * C.kji
    Z.hkj = Z.hkj * A.ii * 2.0 * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 16.0 * np.einsum('hkj,ii,id,idk->hkj', np.einsum('ij,jh,kji->hkj', a, b, c), a, b, c))

    Z.hkj = 2.0 * A.ij * 4.0 * B.jh * C.kji
    Z.hkj *= A.ii * 2.0 * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 16.0 * np.einsum('hkj,ii,id,idk->hkj', np.einsum('ij,jh,kji->hkj', a, b, c), a, b, c))

    Z.hkj = 2.0 * A.ij * 4.0 * B.jh * C.kji
    Z.ijk = Z.hkj * A.ii * 2.0 * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 16.0 * np.einsum('hkj,ii,id,idk->ijk', np.einsum('ij,jh,kji->hkj', a, b, c), a, b, c))

    # Test sum
    Z.ji = A.ij + A.ij
    assert np.allclose(Z.evaluate(A=a), np.einsum('ij->ji', a) + np.einsum('ij->ji', a))

    Z.ji = A.ij + A.ji
    assert np.allclose(Z.evaluate(A=a), np.einsum('ij->ji', a) + np.einsum('ji->ji', a))

    Z.ji = A.ij + 2.0 * B.ji
    assert np.allclose(Z.evaluate(A=a, B=b), np.einsum('ij->ji', a) + 2.0 * np.einsum('ji->ji', b))

    Z.kji = 2.0 * A.ij * 4.0 * B.jh * C.kji * A.ih + A.ij * 2.0 * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 8.0 * np.einsum('ij,jh,kji,ih->kji', a, b, c, a) + 2.0 * np.einsum('ij,id,idk->kji', a, b, c))

    # Test sum itself
    Z.ji = A.ij + A.ji
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij->ji', a) + np.einsum('ij->ij', a))

    Z.ji = A.ij
    Z.ij = Z.ij + A.ij
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij->ji', a) + np.einsum('ij->ij', a))

    Z.ji = A.ij
    Z.ji = Z.ji + A.ij
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij->ji', a) + np.einsum('ij->ji', a))

    Z.kji = A.ij * B.jk
    Z.kij = Z.kij + C.ijk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij,jk->kji', a, b) + np.einsum('ijk->kij', c))

    Z.kji = A.ij * B.jh * C.kji * A.ih
    Z.kij = Z.kij + A.ij * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij,jh,kji,ih->kji', a, b, c, a) + np.einsum('ij,id,idk->kij', a, b, c))

    Z.kji = 2.0 * A.ij * 4.0 * B.jh * C.kji * A.ih
    Z.kij = Z.kij + A.ij * 2.0 * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 8.0 * np.einsum('ij,jh,kji,ih->kji', a, b, c, a) + 2.0 * np.einsum('ij,id,idk->kij', a, b, c))

    Z.kji = 2.0 * A.ij * 4.0 * B.jh * C.kji * A.ih
    Z.kij += A.ij * 2.0 * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 8.0 * np.einsum('ij,jh,kji,ih->kji', a, b, c, a) + 2.0 * np.einsum('ij,id,idk->kij', a, b, c))

    # Test diff
    Z.ji = A.ij - A.ji
    assert np.allclose(Z.evaluate(A=a), np.einsum('ij->ji', a) - np.einsum('ji->ji', a))

    Z.ji = A.ij - 2.0 * B.ji
    assert np.allclose(Z.evaluate(A=a, B=b), np.einsum('ij->ji', a) - 2.0 * np.einsum('ji->ji', b))

    Z.kji = 2.0 * A.ij * 4.0 * B.jh * C.kji * A.ih - A.ij * 2.0 * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 8.0 * np.einsum('ij,jh,kji,ih->kji', a, b, c, a) - 2.0 * np.einsum('ij,id,idk->kji', a, b, c))

    # Test diff itself
    Z.ji = A.ij - A.ji
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij->ji', a) - np.einsum('ij->ij', a))

    Z.ji = A.ij
    Z.ij = Z.ij - A.ij
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij->ji', a) - np.einsum('ij->ij', a))

    Z.ji = A.ij
    Z.ji = Z.ji - A.ij
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij->ji', a) - np.einsum('ij->ji', a))

    Z.kji = A.ij * B.jk
    Z.kij = Z.kij - C.ijk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij,jk->kji', a, b) - np.einsum('ijk->kij', c))

    Z.kji = A.ij * B.jh * C.kji * A.ih
    Z.kij = Z.kij - A.ij * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij,jh,kji,ih->kji', a, b, c, a) - np.einsum('ij,id,idk->kij', a, b, c))

    Z.kji = 2.0 * A.ij * 4.0 * B.jh * C.kji * A.ih
    Z.kij = Z.kij - A.ij * 2.0 * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 8.0 * np.einsum('ij,jh,kji,ih->kji', a, b, c, a) - 2.0 * np.einsum('ij,id,idk->kij', a, b, c))

    Z.kji = 2.0 * A.ij * 4.0 * B.jh * C.kji * A.ih
    Z.kij -= A.ij * 2.0 * B.id * C.idk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 8.0 * np.einsum('ij,jh,kji,ih->kji', a, b, c, a) - 2.0 * np.einsum('ij,id,idk->kij', a, b, c))

    # test neg

    Z.ji = -A.ij
    assert np.allclose(Z.evaluate(A=a), -np.einsum('ij->ji', a))

    Z.ji = -A.ij
    Z.ji -= (-A.ij)
    assert np.allclose(Z.evaluate(A=a), -np.einsum('ij->ji', a) + np.einsum('ij->ji', a))

    # test using subsctipted (implicit) indices
    Z = -A.ij
    assert np.allclose(Z.evaluate(A=a), -np.einsum('ij', a))

    Z = -A.ij
    assert np.allclose(Z.evaluate(A=a), -np.einsum('ij->ij', a))

    Z = A.ij + A.ji
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij', a) + np.einsum('ji', a))

    Z = A.ij + A.ji
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij->ij', a) + np.einsum('ji->ij', a))

    Z = A.ij
    Z = Z.ij + A.ij
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij->ij', a) + np.einsum('ij->ij', a))

    Z = A.ij
    Z = Z.ji + A.ij
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij->ji', a) + np.einsum('ij->ij', a))

    Z = A.ij * B.jk
    Z.ik = Z.ik + C.ijk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij,jk', a, b) + np.einsum('ijk->ik', c))

    Z = A.ij * B.jk
    Z.ki = Z.ik + C.ijk
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), np.einsum('ij,jk->ki', a, b) + np.einsum('ijk->ki', c))

    Z = 2.0 * A.ij * 4.0 * B.jh * C.kji * A.ih
    Z += A.ij * 2.0 * B.id * C.idk * A.jj
    assert np.allclose(Z.evaluate(A=a, B=b, C=c), 8.0 * np.einsum('ij,jh,kji,ih', a, b, c, a) + 2.0 * np.einsum('ij,id,idk,jj', a, b, c, a))

    # Not unique indices
    Z.ii = D.i 
    Z.ii *= A.ij
    assert np.allclose(Z.evaluate(A=a, D=d), np.einsum('i,ij->ij', d, a))

test_eincraft()
