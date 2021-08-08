import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from hmmlearn import hmm
from typing import Tuple, List, Any, Optional, Union


def add_data_splitter(df: pd.DataFrame,
                      proportions: Union[Tuple[float, float],Tuple[float, float, float]]=(0.7,0.05,0.25)
                      )-> pd.DataFrame:

    if len(proportions) == 2:
        t,v = proportions
        t_shape = round(df.shape[0]*t)
        v_shape = df.shape[0] - t_shape
        train = np.full(t_shape,"train")
        val = np.full(v_shape,"val")
        df['data_category'] = np.concatenate((train, val))
    elif len(proportions)==3:
        t,b,v = proportions
        t_shape = round(df.shape[0]*t)
        v_shape = round(df.shape[0]*v)
        b_shape = df.shape[0] - t_shape - v_shape

        train = np.full(t_shape,"train")
        burn = np.full(b_shape,"burn")
        val = np.full(v_shape,"val")        
        df['data_category'] = np.concatenate((train, burn, val))
    else:
        AssertionError("proportions should be lenght-2 or length-3 tuple")
    return df


def add_features(df: pd.DataFrame)-> pd.DataFrame:
    df['ret1d'] =  np.log(df['Close']/df['Close'].shift(1))
    df['dir1d'] = np.sign(df['ret1d'])
    df['high_low_diff'] = (df.High - df.Low)/df.Close
    df['Open_prevClose_diff'] = (df.Open - df['Close'].shift(1))/df['Close'].shift(1)
    df['ret3d'] = df.ret1d.rolling(window=3).mean()
    df['ret5d'] = df.ret1d.rolling(window=5).mean()
    return df


# pca_cols = ['Volume', 'dir1d', 'high_low_diff','Open_prevClose_diff','ret3d','ret5d']
def get_pca_components(df: pd.DataFrame,
                       cols: List[str],
                       n_components: float = 0.9,
                       verbose: bool=False
                       ) -> np.ndarray:
    """
    Run PCA

    """
    X = df[cols]
    X = X.apply(lambda x: (x - x.mean())/x.std())

    for col in cols:
        X[col] = np.maximum(X[col],-2)
        X[col] = np.minimum(X[col],2)

    pca = PCA(n_components=n_components, svd_solver='full')
    X_pca = pca.fit_transform(X)
    if verbose:
        n_pcs= pca.components_.shape[0]
        print("# components:", n_pcs)
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        most_important_names = [cols[most_important[i]] for i in range(n_pcs)]
        dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
        df = pd.DataFrame(dic.items())
        print(df)
        print("Cumulative Explained variance ratio")
        print([round(p,2) for p in pca.explained_variance_ratio_.cumsum()])
    return X_pca  


def run_hmm(input_df: pd.DataFrame,
            trans_mat_diag: Union[List[float], float],
            train_mask: pd.Series = None,
            verbose: bool = False,
            n_regimes: int = None) -> Tuple[int, np.ndarray]:

    if isinstance(trans_mat_diag, float) or len(trans_mat_diag) == 1:
        n_regimes = n_regimes or 2
        d0 = d1 = trans_mat_diag if isinstance(trans_mat_diag, float) else trans_mat_diag[0]
    elif len(trans_mat_diag) == 2:
        n_regimes = len(trans_mat_diag)
        d0, d1 = trans_mat_diag
    else:
        AssertionError("Only # Regimes = 2 is supported")
    
    m = hmm.GaussianHMM(n_components=n_regimes, covariance_type='full', min_covar=0.001, 
                        startprob_prior=1.0, transmat_prior=1.0, means_prior=0, means_weight=0,
                        covars_prior=0.01, covars_weight=1, algorithm='viterbi', random_state=42,
                        n_iter=1000, tol=0.01, verbose=False, params='smc', init_params='smc',)
    m.transmat_ = np.array([[d0, 1-d0],[1-d1, d1]])
    if train_mask is None:
        m.fit(input_df)
    else:
        m.fit(input_df[[train_mask]])

    regimes = m.decode(input_df)[1]
    n_regimes_change = np.minimum(np.abs(regimes[1:] - regimes[:-1]),1).sum()

    if verbose:
        print("--# regimes changed--")
        print(n_regimes_change)
        print("--means--")
        print(pd.DataFrame(m.means_))
        print("--covariance--")
        for i,x in enumerate(m.covars_):
            print(pd.DataFrame(x) )
            if i != len(m.covars_)-1:
                print("---")
        print("--transition matrix--")
        print(pd.DataFrame(m.transmat_))    
        
    return n_regimes_change, regimes

     

