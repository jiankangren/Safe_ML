from sklearn.datasets import load_boston
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from nonconformist.nc import Nc_Reg_Creator
    
boston = load_boston()
idx = np.random.permutation(boston.target.size)

# Divide the data into proper training set, calibration set and test set
idx_train, idx_cal, idx_test = idx[:300], idx[300:399], idx[399:]

model = RandomForestRegressor()	# Create the underlying model
nc_regressor = Nc_Reg_Creator.create_nc(model)	# Create a default nonconformity function
# icp = IcpRegressor(nc)			# Create an inductive conformal regressor

# Fit the ICP using the proper training set
print boston.data[idx_train, :].shape,boston.target[idx_train].shape
nc_regressor.fit(boston.data[idx_train, :], boston.target[idx_train])

# Calibrate the ICP using the calibration set
nc_regressor.calibrate(boston.data[idx_cal, :], boston.target[idx_cal])

# Produce predictions for the test set, with confidence 95%
nc_set=nc_regressor.cal_scores
prediction =nc_regressor.predict(boston.data[idx_test, :],nc_set,significance=0.05)


 # icp.predict(boston.data[idx_test, :], significance=0.05)

# Print the first 5 predictions
print(prediction[:5, :])