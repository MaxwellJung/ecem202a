import numpy as np
import matplotlib.pyplot as plt

def single_pixel_nci(y, c):
    """
    Decode the transport coefficient r for a single pixel.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Pixel intensity over time.
    c : np.ndarray, shape (T,)
        Known temporal code.

    Returns
    -------
    r : float
        Recovered reflectance (strength of code signal at this pixel).
    """

    y = y.astype(np.float32)
    c = c.astype(np.float32)
    if y.max() > 1.0:
        y = y / 255.0

    if len(y) != len(c):
        raise ValueError("Pixel time series and code must have same length")

    numerator = np.dot(c, y)   # sum_t c(t) * y(t)

    denominator = np.dot(c, c) + 1e-8  # sum_t c(t)^2

    r = numerator / denominator

    return r


# Suppose we have 100 frames, assumr l is just 1 for sake of simplicity
T = 100

c = np.random.choice([-1, 1], size=T)

# Simulate a pixel that actually follows the code + noise
true_reflectance = 0.6
y = true_reflectance * c + 0.05 * np.random.randn(T)

# Recover reflectance using NCI
r_est = single_pixel_nci(y, c)

print(f"True reflectance = {true_reflectance:.3f}")
print(f"Estimated reflectance = {r_est:.3f}")

plt.figure(figsize=(8,3))
plt.plot(y, label="Pixel intensity (y)")
plt.plot(c * r_est, label="Recovered code signal (c * r)")
plt.xlabel("Time (frame index)")
plt.legend()
plt.title("Single Pixel Code Correlation")
plt.show()