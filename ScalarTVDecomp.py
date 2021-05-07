# First we load the libraries we need
import time

import numpy as np
import numpy.matlib
from numpy import linalg as LA
import scipy as sp

import scipy.sparse
import scipy.sparse.linalg
import trimesh

class ScalarTVDecomp:
    def __init__(self, mesh, func, components, alpha, alpha_scale):
        self.M = mesh
        self.V = np.array(mesh.vertices)
        self.F = np.array(mesh.faces)
        self.func = func
        self.nComps = components
        self.alpha = alpha
        self.alapa_scale = alpha_scale

    def normest(self, A):
        """
        oli's attempt at normest in matlab
        square root of largest singular value of A?!?
        https://en.wikipedia.org/wiki/Matrix_norm
        """
        [e] = scipy.sparse.linalg.svds(A, k=1, return_singular_vectors=False)
        return e

    def normalize(self, v):
        """
        normalize rows of vectors
        """
        return v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    def adjacency(self, f, n):
        """
        build a sparse adjacency matrix from faces in coordinate format.
        faces have some redundancy which we use a bitmask to filter.
        """
        A = sp.sparse.dok_matrix((n, n), dtype=bool)
        i, j, k = f.T
        A[i, j] = True
        A[j, k] = True
        A[k, i] = True
        A = A.tocsr()
        A = A + A.T
        return A

    def edges(self, f, n):
        """
        build an adjacency matrix, return edges in only one direction (upper triangular)
        """
        A = sp.sparse.triu(self.adjacency(self, f, n))
        ei, ej = A.nonzero()
        ed = np.vstack([ei, ej]).T
        # sort for consistency
        return np.sort(ed, axis=0)

    def face_normals(self, v, f):
        """
        compute normals of faces f
        """
        i, j, k = f.T
        a, b, c = v[i], v[j], v[k]

        # Compute edges, ensuring correct winding order
        ei = b - c
        ek = a - b

        # Compute the face normals
        n = np.cross(ek, ei, axis=1)

        return self.normalize(n)

    def triangle_area(self, v, f):
        """
        compute triangle areas
        """
        i, j, k = f.T
        a, b, c = v[i], v[j], v[k]

        ac = c - a
        bc = c - b

        return np.linalg.norm(np.cross(ac, bc, axis=1), axis=1) / 2

    def vertex_area(self, v, f):
        """
        compute total area about vertices
        3.59ms for 281,724 faces, not bad son
        """
        n = len(v)
        A = np.zeros((3, len(f)))

        area = self.triangle_area(v, f)

        # set internal angles at vertex location in face array
        # using indexes that have duplicate values to increment doesn't work
        A[0] = area
        A[1] = area
        A[2] = area

        # some esoteric numpy for summing at duplicated indices
        # coo matrices are also an option
        data = A.ravel()
        cols = f.T.ravel()

        M = np.zeros(n)
        np.add.at(M, cols, data)

        return sp.sparse.diags(M)

    def barycentric_mass(self, v, f):
        return self.vertex_area(v, f) / 3

    def div(self, v, f):
        G = self.grad(v, f)
        A = self.barycentric_mass(v, f)
        area = self.triangle_area(v, f)
        TA = scipy.sparse.diags(np.tile(area, 3))
        D = -sp.sparse.diags(1 / A.diagonal()) @ G.T @ TA
        return D

    def grad(self, v, f):
        # G_F is a diagonal matrix with the areas of the faces along the diagonal, repeated three times
        area = self.triangle_area(v, f)

        # Preprocess by computing 1/(2*G)
        G_F = scipy.sparse.diags(1 / np.tile(2 * area, 3))

        # To build E:
        # Get the edges opposite each vertex
        # Vectors should form a closed chain and sum to zero in a counterclockwise direction
        i, j, k = f.T
        a, b, c = v[i], v[j], v[k]

        ei = b - c
        ej = c - a
        ek = a - b

        # Compute the face normals
        n = self.normalize(np.cross(ek, ei, axis=1))

        # Rotate each edge by 90 degrees s.t. it points inward
        # Rotation is just the cross product with normal
        ei = np.cross(ei, n, axis=1)
        ej = np.cross(ej, n, axis=1)
        ek = np.cross(ek, n, axis=1)

        # Fill up the matrix E, which is 3|F| x V
        r = np.arange(len(f) * 3)
        rows = np.hstack([r, r, r])
        cols = np.hstack([i, i, i, j, j, j, k, k, k])
        data = np.hstack([
            ei[:, 0], ei[:, 1], ei[:, 2],
            ej[:, 0], ej[:, 1], ej[:, 2],
            ek[:, 0], ek[:, 1], ek[:, 2],
        ])

        E = scipy.sparse.coo_matrix((data, (rows, cols)))

        # Grad is G_F x E
        return G_F @ E

    def pdhg_scalar(self, prox_f_star, prox_g, gradOP, divOP, u, q, gamma, sigma, tau, max_iters):
        # PDHG_ACC: implements the accelerated version of the Primal Dual Hybird
        # Gradient of: A.Chambolle, T.Pock "A first-order primal-dual algorithm for convex problems
        # with applications to imaging" (https://hal.archives-ouvertes.fr/hal-00490826/document)
        i = 1
        err = np.inf
        u_bar = u
        err_thresh = 1e-5
        u_old = u

        while (i < max_iters and err > err_thresh):
            # update the primal and dual variables
            q_old = q

            f_data = q + (sigma * (gradOP @ u_bar))
            q = prox_f_star(f_data, sigma)

            g_data = u + (tau * (divOP @ q))
            u = prox_g(g_data, tau)
            # update step parameters
            theta = 1 / np.sqrt(1 + 2 * gamma * tau)
            tau = tau * theta
            sigma = sigma / theta

            # extrapolate
            u_bar = u + theta * (u - u_old)
            err = LA.norm(u - u_old)

            i = i + 1
            if (i % 30 == 0):
                u_old = u
        print("Iterations: ", i, " | Error: ", err)
        return u, q

    def decompose_scalar(self):
        """
            decomposeScalar: apply the spectral TV decomposition to a scalar signal f defined on the vertices of the
            domain M, applying the algorithm 3 in the paper.

            # decompose Scalar function
            # Input
                 VERTS :: vertices
                 FACES :: faces
                 f :: scalar value function vector [|V| x 1]
                 d :: the number of spectral components to be computed
                 alpha :: maximum diffusion time
                 alpha_scale :: a constant used to scale alpha
            # Returns:
                 phi :: the eigen components

        """
        VERTS = self.V
        FACES = self.F
        f = self.func
        d = self.nComps
        alpha = self.alpha
        alpha_scale = self.alapa_scale

        # build the gradient and divergence operators on surfaces of TV
        gradOP = self.grad(VERTS, FACES)
        divOP = self.div(VERTS, FACES)

        # get the timestep values through the norm
        pt = self.normest(gradOP)

        # Spectral decomposition variable initializations

        dim_F = FACES.shape[0]
        dim_V = VERTS.shape[0]
        vt = np.zeros((dim_V, 1))
        v = np.zeros((dim_V, 1))
        f = np.reshape(f, (dim_V, 1))

        # initialize the primal signal u as the mean vector of f
        u = np.ones((dim_V, 1)) * np.mean(f)
        # initialize the dual signal q
        dim_gradOP = gradOP.shape[0]
        q = np.zeros((dim_gradOP, 1))

        # initialize spectral components
        sub = np.zeros((dim_V, d))
        old_R = np.zeros((dim_V, d))

        # setting proximity operators
        def prox_F_star(data, param):
            # proximity F star operator for scalar decomposition
            temp = np.reshape(data, (-1, 3), order='F')
            temp = np.square(temp)
            temp = np.sqrt(np.sum(temp, 1))
            temp = np.matlib.repmat(temp, 1, 3)
            temp = np.maximum(np.ones_like(temp), temp)
            temp = data / temp.T
            return temp

        def prox_g(x, tau):
            p = tau / alpha
            y = f + v
            return (x + (p * y)) / (1 + p)

        max_iter = 3000;

        print("STARTING THE SPECTRAL DECOMPOSITION TING 💩")

        for i in range(d):
            # rescale the maximum number of operations to speed up the algorithm
            max_iter = max(max_iter * 0.95, 100)

            # print(u.shape)
            sub[:, i] = np.squeeze(u)

            # compute the solution to \min_u TV(u) + 1/(2*alpha) ||u-(f+v)||^2 via
            # the PDHG algorithm
            [u, q] = self.pdhg_scalar(prox_F_star, prox_g, gradOP, divOP, u, q, 0.7 / alpha, 1 / pt, 1 / pt, max_iter)

            alpha_old = alpha

            # scale alpha  for uniform timescale
            # alpha = alpha * alpha_scale
            alpha = max(alpha - alpha_scale, 1e-6);
            v = (v + (f - u)) * alpha / alpha_old;

        phi = sub.copy()
        for n in range(1, phi.shape[1]):
            phi[:, n] = phi[:, n] - sub[:, n - 1]

        return phi

    def GetTVSignal(self, phi):
        A = self.barycentric_mass(self.V, self.F)
        S = np.sum(np.abs(A @ phi), 0)
        return S

