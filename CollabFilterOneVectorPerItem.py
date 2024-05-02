'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib
import numpy as np

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=ag_np.ones(n_users), # FIX dimensionality
            c_per_item=ag_np.ones(n_items), # FIX dimensionality
            U=0.01 * random_state.randn(n_users, self.n_factors), # FIX dimensionality
            V=0.01 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # print("user_id_N shape:", user_id_N.shape)
        # print("item_id_N shape:", item_id_N.shape)
        # TODO: Update with actual prediction logic
        N = user_id_N.size
        yhat_N = ag_np.ones(N)
        user_biases = b_per_user[user_id_N]
        item_biases = c_per_item[item_id_N]
        user_factor_items = U[user_id_N]#[:,:self.n_factors]
        item_factor_items = V[item_id_N]#[:,:self.n_factors]
        # ag_np.set_printoptions(threshold=ag_np.inf, linewidth=ag_np.inf)
        # if ag_np.isnan(user_biases).any():
        #     print("user biases:")
        #     print(user_biases)
        # if ag_np.isnan(item_biases).any():
        #     print("item biases:")
        #     print(item_biases)
        # print("user factors:")
        # print(user_factor_items)
        # print("item factors:")
        # print(item_factor_items)
        factor_mult = user_factor_items * item_factor_items
        factor_sum = ag_np.sum(factor_mult, axis=1)
        yhat_N = mu + user_biases + item_biases + factor_sum
        if ag_np.isnan(yhat_N).any():
            print("yhat_N:")
            print(yhat_N)
        # for i in range(N):
        #     user_index = int(user_id_N[i])
        #     item_index = int(item_id_N[i])
            
        #     user_bias = b_per_user[user_index] if user_index in b_per_user else 0
        #     item_bias = c_per_item[item_index] if item_index in c_per_item else 0
        #     user_factors = U[user_index]
        #     item_factors = V[item_index]
        #     np_user_factors = np.asarray(user_factors._value if hasattr(user_factors, "_value") else user_factors)
        #     np_item_factors = np.asarray(item_factors._value if hasattr(item_factors, "_value") else item_factors)
        #     # print("Numpy user factors:")
        #     # print(np_user_factors)
        #     factors_sum = np.sum(np_user_factors * np_item_factors)
        #     yhat = mu + user_bias + item_bias + factors_sum
        #     if hasattr(yhat, "_value"):
        #         # print("yhat:")
        #         # print(yhat)
        #         yhat = yhat._value[0]
        #         print("yhat value:")
        #         print(yhat)
            
        #     yhat_N[i] = yhat
            
        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        user_id_N, item_id_N, y_N = data_tuple
        b_per_user = param_dict['b_per_user']
        c_per_item = param_dict['c_per_item']
        user_biases = b_per_user[user_id_N]
        item_biases = c_per_item[item_id_N]
        U = param_dict['U']
        V = param_dict['V']
        user_factors = U[user_id_N]
        item_factors = V[item_id_N]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        # loss = ag_np.sum((y_N - yhat_N)**2)
        # # Small optimization to avoid doing unnecessary work when alpha is 0
        # if self.alpha != 0:
        #     loss = loss + self.alpha * (
        #         ag_np.sum(user_biases**2) + ag_np.sum(item_biases**2) +
        #             ag_np.sum(user_factors**2) +
        #             ag_np.sum(item_factors**2)
        loss = ag_np.sum((y_N - yhat_N)**2) + self.alpha * (
            ag_np.sum(user_biases**2) + ag_np.sum(item_biases**2) +
                ag_np.sum(user_factors**2) +
                ag_np.sum(item_factors**2)
        )
        return loss


if __name__ == '__main__':
    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    # alpha = 0.0
    alpha = 0.3
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=10, alpha=alpha)
    model.init_parameter_dict(n_users, n_items, train_tuple)


    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)
