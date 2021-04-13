def curvature_matrix_from_blurred_mapping_matrix_jit(
    self, blurred_mapping, noise_vector
):
    flist = np.zeros(blurred_mapping.shape[0])
    iflist = np.zeros(blurred_mapping.shape[0], dtype="int")
    return self.curvature_matrix_from_blurred_mapping_jitted(
        blurred_mapping, noise_vector, flist, iflist
    )


@staticmethod
@numba.jit(nopython=True)
def curvature_matrix_from_blurred_mapping_jitted(
    blurred_mapping, noise_vector, flist, iflist
):
    mapping_shape = blurred_mapping.shape

    covariance_matrix = np.zeros((mapping_shape[1], mapping_shape[1]))

    for image_index in range(mapping_shape[0]):
        index = 0
        for pix_index in range(mapping_shape[1]):
            if blurred_mapping[image_index, pix_index] > 0.0:
                index += 1
                flist[index] = (
                    blurred_mapping[image_index, pix_index] / noise_vector[image_index]
                )
                iflist[index] = pix_index

        if index > 0:
            for i1 in range(index + 1):
                for j1 in range(index + 1):
                    ix = iflist[i1]
                    iy = iflist[j1]
                    covariance_matrix[ix, iy] += flist[i1] * flist[j1]

    for i in range(mapping_shape[1]):
        for j in range(mapping_shape[1]):
            covariance_matrix[i, j] = covariance_matrix[j, i]

    return covariance_matrix
