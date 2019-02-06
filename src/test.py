
import numpy as np

scores = []
part = np.array([1.0,2.0,3.0,1.0,4.5,3.0,1.0,1.5,2.0,2.5]).reshape((5,2))
kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))   # (5,1000)
print(kl)
print(np.sum(kl, 1))
kl = np.mean(np.sum(kl, 1)) #scalar value
print(kl)
scores.append(np.exp(kl))