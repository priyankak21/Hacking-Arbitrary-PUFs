import numpy as np
from sklearn.svm import LinearSVC

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def re_encode_challenge(C):     ## reencodes a {0,1} challenge to {1,-1} challenge
    return 1-2*C

def calculate_1D_X(D):          ## calculates the xi variables corresponidng to each challenge row 
    X = np.zeros_like(D)
    for i in range(len(D)):
        prod = 1
        for j in range(len(D[i])):
            prod=prod*D[i,j]
        
        X[i,0] = prod 
        for j in range(len(D[i])-1):
            X[i,j+1] = X[i,j]/D[i,j]
    return X

def get_2D_X(X):			## 2d products xi*xj for i!= j
    X2 = np.zeros( (len(X),int((len(X[0]))*( len(X[0]) -1)/2)) )
    for i in range(len(X2)):
        lis = []
        for j in range(len(X[i])):
            for k in range(j+1,len(X[i])):
                lis.append(X[i,j]*X[i,k])

        X2[i] = np.array(lis)
    return X2

def join(X,X2):		## join one dimensional and squared variables to get the feature 
    return np.concatenate((X,X2),axis = 1)


################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
    features = my_map(X_train)
    clf = LinearSVC(dual = False,C = 1)
    
    clf.fit(features,y_train)
    b = clf.intercept_[0]
    w = clf.coef_[0]
 
    return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	D = re_encode_challenge(X)
	X1 = calculate_1D_X(D)
	X2 = get_2D_X(X1)
	feat = join(X1,X2)
	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
	return feat