import numpy as np
from pyramid import build_pyr, collapse_pyr
from patchify import patchify

def dist(x1,x2,y1,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def compute_optimal_transport(X, Y, iterations = 100, projections = 1, return_asg = False):
    """
    Compute the optimal transport between two point clouds.

    Parameters:
    - X (ndarray): Source point cloud
    - Y (ndarray): Target point cloud
    - iterations (int): Number of iterations for the optimization process
    - projections (int): Number of random projections used in each iteration
    - return_asg (bool): If True, return the sorted indices of the final point cloud

    Returns:
    - ndarray: Optimal transport between the source and target point clouds
    - (Optional) ndarray: Sorted indices of the final point cloud along with the source and target point clouds
    """
    
    Z = X.copy()
    N = X.shape[0]
    D = X.shape[-1]
    lr = 1
    if(projections == -1):
        projections = D
    
    for i in range(iterations):
        grad = 0
        for p in range(projections):
            theta = np.random.uniform(-1, 1, D)
            theta /= np.linalg.norm(theta)
            proj_z = Z @ theta
            proj_y = Y @ theta
            sz = np.argsort(proj_z)
            sy = np.argsort(proj_y)
            grad += proj_z[sz] - proj_y[sy]
        grad = grad.reshape(-1, 1) / projections
        grad = grad @ theta.reshape(-1, 1).T
        Z[sz] -= lr * grad
        lr *= 0.999
    if(return_asg):
        return Z, sy, sz
    return Z

def compute_optimal_transport_barycenter(X_ini, lambdas, point_clouds, iterations = 1000, projections = 1, return_asg = False):
    """
    Compute the optimal transport barycenter of multiple point clouds.

    Parameters:
    - X_ini (ndarray): Initial point cloud for the barycenter computation
    - lambdas (list): List of weights corresponding to the contribution of each point cloud
    - point_clouds (list): List of point clouds (ndarrays) to be averaged
    - iterations (int): Number of iterations for the optimization process
    - projections (int): Number of random projections used in each iteration
    - return_asg (bool): If True, return the sorted indices of the final point cloud

    Returns:
    - ndarray: Optimal transport barycenter of the input point clouds
    - (Optional) ndarray: Sorted indices of the final point cloud along with the input point clouds
    """
    
    assert 1 - 1e-5 < sum(lambdas) < 1 + 1e-5, 'the sum of the weights must be 1'
    Z = X_ini.copy()
    N = X_ini.shape[0]
    D = X_ini.shape[-1]
    lr = 1
    if(projections == -1):
        projections = D
    
    for i in range(iterations):
        grad = np.zeros((N, D))
        for p in range(projections):
            theta = np.random.uniform(-1, 1, D)
            theta /= np.linalg.norm(theta)
            for lmb in range(len(lambdas)):
                grad_tmp = 0
                proj_z = Z @ theta
                proj_y = point_clouds[lmb] @ theta
                sz = np.argsort(proj_z)
                sy = np.argsort(proj_y)
                grad_tmp += lambdas[lmb] * (proj_z[sz] - proj_y[sy])
                grad_tmp = grad_tmp.reshape(-1, 1)
                grad += grad_tmp @ theta.reshape(-1, 1).T
        Z[sz] -= lr * grad / projections
        lr *= 0.999
    if(return_asg):
        return Z, sy, sz
    return Z

def compute_optimal_assignment(X, Y):
    """
    Compute the optimal assignment between two sets of points by sorting them.

    Parameters:
    - X (ndarray): Input matrix representing the source set of points
    - Y (ndarray): Input matrix representing the target set of points

    Returns:
    - ndarray: The optimal assignment from X to Y
    """
    
    Z = np.zeros_like(X)
    for i in range(X.shape[-1]):
        sz = X.reshape(-1,X.shape[-1])[:, i].argsort()
        sy = Y.reshape(-1,X.shape[-1])[:, i].argsort()
        Z[sz, i] = Y.reshape(-1, X.shape[-1])[sy, i]
    return Z

def texture_mixing(textures, lambdas, noise=None, scales=4, orientations=4, patch=0, patch_size=4, step=2,
                   spectrum=False, gromov=False, n_iter=100):
    """
    Perform texture mixing.

    Parameters:
    - textures (list of ndarray): List of input textures (numpy arrays)
    - lambdas (list): List of weights for each texture in the mixing process, this must sum to 1
    - noise (ndarray, optional): Input noise image. If not provided, a random noise image is generated
    - scales (int, optional): Number of scales for the image pyramid
    - orientations (int, optional): Number of orientations for the image pyramid
    - patch (int, optional): Type of patches to use (0 = none, 1 = non-overlapping, 2 = overlapping)
    - patch_size (int, optional): Size of the patches
    - step (int, optional): Step size for patch extraction and reconstruction
    - spectrum (bool, optional): Apply spectrum constraint during mixing
    - gromov (bool, optional): Apply Gromov-Wasserstein constraint during patch reconstruction at the last scale of the pyramid
    - n_iter (int, optional): Number of iterations for the mixing process

    Returns:
    - ndarray: Resulting mixed texture image.

    Note:
    The function performs texture mixing by iteratively optimizing the transport barycenter of the input textures
    and reconstructing the noisy image using optimal transport.
    """
    
    assert 1 - 1e-5 < sum(lambdas) < 1 + 1e-5, 'the sum of the weights must be 1'
    size = textures[0].shape[0]

    if noise is None:
        noise = np.random.randn(size, size, 3)

    true_bary = compute_optimal_transport_barycenter(noise.reshape(-1, 3), lambdas,
                                                     [x.reshape(-1, 3) for x in textures], iterations=100).reshape(
        size, size, 3)

    pyramids = np.empty(len(textures) + 1, dtype=object)

    for i in range(len(textures)):
        _, b = build_pyr(textures[i], scales, orientations, cplx=0)
        pyramids[i] = b

    _, b = build_pyr(noise, scales, orientations, cplx=0)
    pyramids[i + 1] = b

    bary = np.zeros_like(b)

    if patch:
        patches = np.zeros_like(b)
        q = patch_size ** 2 * 3

    for i in range(len(b)):
        n = b[i].shape[0]

        bary[i] = compute_optimal_transport_barycenter(pyramids[-1][i].reshape(-1, 3), lambdas,
                                                       [pyramids[x][i].reshape(-1, 3) for x in range(len(textures))],
                                                       iterations=100).reshape(n, n, 3)

        if patch:
            if patch == 1:
                noise_patches = np.transpose(
                    pyramids[-1][i].reshape(n // patch_size, patch_size, n // patch_size, patch_size, 3),
                    (0, 2, 1, 3, 4)).reshape(-1, q)
                patches[i] = compute_optimal_transport_barycenter(noise_patches, lambdas,
                                                                  [np.transpose(
                                                                      pyramids[x][i].reshape(n // patch_size, patch_size,
                                                                                                n // patch_size,
                                                                                                patch_size, 3),
                                                                      (0, 2, 1, 3, 4)).reshape(-1, q) for x in
                                                                   range(len(textures))],
                                                                  iterations=50)
            else:
                noise_patches = patchify(pyramids[-1][i], patch_size=(patch_size, patch_size, 3),
                                         step=(step, step, 1))[:, :, 0].reshape(-1, q)
                patches[i] = compute_optimal_transport_barycenter(noise_patches, lambdas,
                                                                  [patchify(pyramids[x][i], patch_size=(patch_size,
                                                                                                         patch_size, 3),
                                                                            step=(step, step, 1))[:, :, 0].reshape(-1, q) for
                                                                   x in range(len(textures))],
                                                                  iterations=50)

    if spectrum:
        noise = spectrum_constraint(noise, true_bary)

    for i in range(n_iter):
        _, noisy_b = build_pyr(noise, scales, orientations, cplx=0)

        for l in range(len(noisy_b)):
            n = noisy_b[l].shape[0]

            Z = compute_optimal_assignment(noisy_b[l].reshape(-1, 3), bary[l].reshape(-1, 3)).reshape(n, n, 3)

            if patch:
                if patch == 1:
                    Z = np.transpose(Z.reshape(n // patch_size, patch_size, n // patch_size, patch_size, 3),
                                     (0, 2, 1, 3, 4)).reshape(-1, q)
                    Z = compute_optimal_transport(Z.reshape(-1, q), patches[l].reshape(-1, q), iterations=100)
                    Z = Z.reshape(n // patch_size, n // patch_size, patch_size, patch_size, 3)

                    if gromov and l == len(noisy_b) - 1:
                        nbr = i % 4

                        for k in range(1, 16, 4):
                            if k < 10:
                                continue

                            previous_size = noisy_b[k].shape[0]
                            size_k = noisy_b[k].shape[0] // patch_size

                            p1 = np.transpose(noisy_b[k].reshape(size_k, patch_size, size_k, patch_size, 3),
                                              (0, 2, 1, 3, 4)).reshape(-1, patch_size ** 2, 3)
                            p2 = np.transpose(noisy_b[k + 1].reshape(size_k, patch_size, size_k, patch_size, 3),
                                              (0, 2, 1, 3, 4)).reshape(-1, patch_size ** 2, 3)
                            p3 = np.transpose(noisy_b[k + 2].reshape(size_k, patch_size, size_k, patch_size, 3),
                                              (0, 2, 1, 3, 4)).reshape(-1, patch_size ** 2, 3)
                            p4 = np.transpose(noisy_b[k + 3].reshape(size_k, patch_size, size_k, patch_size, 3),
                                              (0, 2, 1, 3, 4)).reshape(-1, patch_size ** 2, 3)
                            rec1 = np.zeros((size_k ** 2, patch_size ** 2, 3))
                            rec2 = np.zeros((size_k ** 2, patch_size ** 2, 3))
                            rec3 = np.zeros((size_k ** 2, patch_size ** 2, 3))
                            rec4 = np.zeros((size_k ** 2, patch_size ** 2, 3))
                            Ck = np.zeros((4, patch_size ** 2, patch_size ** 2))

                            for nb_pt in range(0, size_k ** 2):
                                Ck[0] = cdist(p1[nb_pt].reshape(-1, 3), p1[nb_pt].reshape(-1, 3))
                                Ck[1] = cdist(p2[nb_pt].reshape(-1, 3), p2[nb_pt].reshape(-1, 3))
                                Ck[2] = cdist(p3[nb_pt].reshape(-1, 3), p3[nb_pt].reshape(-1, 3))
                                Ck[3] = cdist(p4[nb_pt].reshape(-1, 3), p4[nb_pt].reshape(-1, 3))

                                gw1 = ot.gromov.gromov_wasserstein(
                                    Ck[0], Ck[nbr], np.full(patch_size ** 2, 1 / (patch_size ** 2)),
                                    np.full(patch_size ** 2, 1 / (patch_size ** 2)), 'square_loss', verbose=False)

                                gw2 = ot.gromov.gromov_wasserstein(
                                    Ck[1], Ck[nbr], np.full(patch_size ** 2, 1 / (patch_size ** 2)),
                                    np.full(patch_size ** 2, 1 / (patch_size ** 2)), 'square_loss', verbose=False)

                                gw3 = ot.gromov.gromov_wasserstein(
                                    Ck[2], Ck[nbr], np.full(patch_size ** 2, 1 / (patch_size ** 2)),
                                    np.full(patch_size ** 2, 1 / (patch_size ** 2)), 'square_loss', verbose=False)

                                gw4 = ot.gromov.gromov_wasserstein(
                                    Ck[3], Ck[nbr], np.full(patch_size ** 2, 1 / (patch_size ** 2)),
                                    np.full(patch_size ** 2, 1 / (patch_size ** 2)), 'square_loss', verbose=False)

                                rec1[nb_pt, gw1.argmax(axis=1)] = p1[nb_pt].reshape(-1, 3)
                                rec2[nb_pt, gw2.argmax(axis=1)] = p2[nb_pt].reshape(-1, 3)
                                rec3[nb_pt, gw3.argmax(axis=1)] = p3[nb_pt].reshape(-1, 3)
                                rec4[nb_pt, gw4.argmax(axis=1)] = p4[nb_pt].reshape(-1, 3)

                            p1 = rec1.reshape(size_k, size_k, patch_size, patch_size, 3)
                            p2 = rec2.reshape(size_k, size_k, patch_size, patch_size, 3)
                            p3 = rec3.reshape(size_k, size_k, patch_size, patch_size, 3)
                            p4 = rec4.reshape(size_k, size_k, patch_size, patch_size, 3)

                            p1 = np.transpose(p1, ((0, 2, 1, 3, 4))).reshape(previous_size, previous_size, 3)
                            p2 = np.transpose(p2, ((0, 2, 1, 3, 4))).reshape(previous_size, previous_size, 3)
                            p3 = np.transpose(p3, ((0, 2, 1, 3, 4))).reshape(previous_size, previous_size, 3)
                            p4 = np.transpose(p4, ((0, 2, 1, 3, 4))).reshape(previous_size, previous_size, 3)

                            noisy_b[k] = p1
                            noisy_b[k + 1] = p2
                            noisy_b[k + 2] = p3
                            noisy_b[k + 3] = p4

                            Z = np.transpose(Z, ((0, 2, 1, 3, 4))).reshape(n, n, 3)
                        else:
                            Z = patchify(Z, patch_size=(patch_size, patch_size, 3), step=(step, step, 1))[:, :, 0]
                            previous_shape = Z.shape
                            Z = compute_optimal_transport(Z.reshape(-1, q), patches[l].reshape(-1, q),
                                                          iterations=100)
                            Z = Z.reshape(previous_shape)

                            overlap_factors = np.zeros((n, n, 3))
                            for y in range(0, n - step, step):
                                for x in range(0, n - step, step):
                                    overlap_factors[y:y + patch_size, x:x + patch_size] += 1

                            reconstructed = np.zeros((n, n, 3))
                            for y in range(0, previous_shape[0]):
                                for x in range(0, previous_shape[1]):
                                    h = y * step
                                    w = x * step
                                    reconstructed[h:h + patch_size, w:w + patch_size] += Z[y, x]

                            Z = reconstructed / overlap_factors

                    noisy_b[l] = Z

            noisy_b_r = np.zeros_like(noisy_b)
            noisy_b_g = np.zeros_like(noisy_b)
            noisy_b_b = np.zeros_like(noisy_b)

            for c in range(len(noisy_b_r)):
                noisy_b_r[c] = noisy_b[c][:, :, 0]
                noisy_b_g[c] = noisy_b[c][:, :, 1]
                noisy_b_b[c] = noisy_b[c][:, :, 2]

            reconstructed_r = collapse_pyr(noisy_b_r, scales, orientations)
            reconstructed_g = collapse_pyr(noisy_b_g, scales, orientations)
            reconstructed_b = collapse_pyr(noisy_b_b, scales, orientations)

            reconstructed = np.concatenate((reconstructed_r[:, :, np.newaxis], reconstructed_g[:, :, np.newaxis],
                                            reconstructed_b[:, :, np.newaxis]), -1)

            noise = reconstructed.reshape(size, size, 3)
            matched = compute_optimal_transport(noise.reshape(-1, 3), true_bary.reshape(-1, 3),
                                                iterations=50).reshape(size, size, 3)

            noise = matched

    return matched



def spectrum_constraint(X, Y):
    """
    Code adapted from MATLAB: Tartavel, G., Gousseau, Y., & Peyré, G. (2015). Variational texture synthesis with sparsity and spectrum constraints. Journal of Mathematical Imaging and Vision, 52, 124-144. to match the spectrum of an image Y to X.
    Match the spectrum of an image Y to that of image X using FFT-based processing.

    Parameters:
    - X (ndarray): Reference image.
    - Y (ndarray): Target image.

    Returns:
    - ndarray: Image with the spectrum matched to that of the reference image.
    """
    
    # Compute FFT
    ft_x = np.fft.fft2(X)
    ft_y = np.fft.fft2(Y)
    
    # Compute new image FT
    innerProd = np.sum(ft_x * np.conj(ft_y), axis = -1)
    dephase = innerProd / (np.abs(innerProd) + 1e-7)
    ft_z = ft_y * dephase[:, :, np.newaxis]
    
    # Inverse FFT
    Z = np.fft.ifft2(ft_z).real
    
    #E = np.sum((X.flatten() - Z.flatten())**2)
    
    return Z
