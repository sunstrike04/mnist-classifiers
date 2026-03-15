import numpy as np


def pool(x, pool_size=7):
    """
    Implement the pooling featurizer.
    Args:
        x: n x d matrix with a d-dimensional feature for each of the n points
        pool_size: size of the pooling window
    Returns:
        feat: n x dd matrix with the pooled features for each of the n points
            dd = d // pool_size ** 2
    """
    # BEGIN YOUR CODE
    n, d = x.shape
    h = w = int(np.sqrt(d)) # Assumes square images, 28x28 for MNIST
    x_reshaped = x.reshape(n, h, w)

    # Calculate dimensions of the pooled output
    pooled_h, pooled_w = h // pool_size, w // pool_size

    # Reshape and take the mean over the pooling windows
    feat = x_reshaped.reshape(n, pooled_h, pool_size, pooled_w, pool_size).mean(axis=(2, 4))
    return feat.reshape(n, -1)
    # END YOUR CODE


def hog(x, pool_size=7, angle_bins=18):
    """
    Implement the Histogram of Gradient featurizer.
    Args:
        x: n x d matrix with a d-dimensional feature for each of the n points
        pool_size: size of the pooling window
        angle_bins: number of bins to use for the angle histogram
            For example, if angle_bins=18, then you should split the gradient
            orientation into 18 equal bins between 0 and 360, each one spanning
            20 degrees.
    Returns:
        feat: n x dd matrix with the HOG features for each of the n points
            dd = d // pool_size ** 2 * angle_bins
    """
    x = x.reshape(-1, 28, 28)
    # BEGIN YOUR CODE
    n, h, w = x.shape

    # 1. Compute gradients
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :, :-1] = np.diff(x, axis=2)
    gy[:, :-1, :] = np.diff(x, axis=1)

    # 2. Compute magnitude and orientation
    mag = np.sqrt(gx**2 + gy**2)
    ori = np.arctan2(gy, gx) * (180 / np.pi) + 180 # 0-360 degrees

    # 3. Bin orientations
    bin_width = 360 / angle_bins
    bins = (ori // bin_width).astype(int)
    bins[bins >= angle_bins] = angle_bins - 1 # Handle edge case for 360 degrees

    # 4. Compute pooled histograms
    pooled_h, pooled_w = h // pool_size, w // pool_size
    num_pools = pooled_h * pooled_w
    feat = np.zeros((n, pooled_h, pooled_w, angle_bins))

    for i in range(pooled_h):
        for j in range(pooled_w):
            # Get the window for the current pool
            mag_win = mag[:, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            bin_win = bins[:, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]

            # Use np.bincount for each image in the batch
            for img_idx in range(n):
                hist = np.bincount(bin_win[img_idx].ravel(), weights=mag_win[img_idx].ravel(), minlength=angle_bins)
                feat[img_idx, i, j, :] = hist

    return feat.reshape(n, -1)
    # END YOUR CODE


def featurize(x, type='raw', pool_size=7, angle_bins=18):
    if type == 'raw':
        x = x.reshape(x.shape[0], -1) - 0.5
    elif type == 'pool':
        x = pool(x, pool_size=pool_size)
    elif type == 'hog':
        x = hog(x, pool_size=pool_size, angle_bins=angle_bins)
    return x
