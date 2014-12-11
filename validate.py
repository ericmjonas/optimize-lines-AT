import numpy as np
import util
import axfunc

"""
basic testing code -- because this explicitly instantiates the giant
matrix A, it consumes a tremendous amount of memory. Therefor
we only compare with small images and small atomic sets

"""


sc = util.create_iter_scene_config(32, 32,  10)
ac = util.create_real_atom_config(sc, XGRID_N=40, Y_GRID=40,  ZGRID_N=10)

image = np.random.rand(sc['uy'][2],
                       sc['ux'][2],
                       sc['y'][2],
                       sc['x'][2]).astype(np.float32)


am = util.create_atom_matrix_iter(ac, sc)
print am.shape, image.shape

y_exact = np.dot(am.T, image.flatten())

y_nc = axfunc.compute_ATx(ac, sc, image)

np.testing.assert_allclose(y_exact, y_nc,
                           rtol=1e-5, atol=1e-9)

