from feature.eddp_feature import triplet_feature, twobody_feature
# import eddp_feature
import numpy as np

pos = np.random.uniform(-2., 2., (10, 3))
exponents1 = np.array([1,2,3], dtype=np.float)
exponents2 = np.array([1,2,3], dtype=np.float)
radius = 1.0
cell = np.eye(3, dtype=np.float) * 1.5
sup_x = np.array([-1, 2], dtype=np.int32)
sup_y = np.array([-1, 2], dtype=np.int32)
sup_z = np.array([-1, 2], dtype=np.int32)
print(pos)
triple_feature = triplet_feature(pos, exponents1, exponents2, radius, cell, sup_x, sup_y, sup_z)
twobody_feature = twobody_feature(pos, exponents1, radius, cell, sup_x, sup_y, sup_z)

print(triple_feature.shape)
print(twobody_feature.shape)