# %%

import numpy as np



array1=np.zeros((2,10))
array2=np.ones((2,10))
array2[:,:5]*=-1
res=np.linalg.norm(array1-array2,ord=1)
print(array1-array2)
print(res)
# %%
0