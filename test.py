# %%
import numpy as np

# %%
a = np.random.randn(3, 3)  

# %%
u, s, vh = np.linalg.svd(a, full_matrices=True)
# %%
print(u.T@u)
# %%
print(u[0]@u.T)
# %%
