import numpy as np

class ConservationCorrectedANN:
    """Development model: original ANN plus training-selected soft projection.

    Input columns: Ca_i, Ca_o, Ca_m, De_m, n, log(2*v_m/v_i).
    Equal 50-micrometre inner and individual middle inlet widths are assumed.
    Predicted areas use units of 10^-9 m^2. No extrapolation guarantee.
    """
    def __init__(self,inner,shell,variances,tau=.4):
        self.inner=inner;self.shell=shell;self.variances=np.asarray(variances,float);self.tau=tau
    def predict(self,X):
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=6 or not np.isfinite(X).all():
            raise ValueError('Expected six finite columns, ending in log(2vm/vi)')
        p=np.column_stack([self.inner.predict(X[:,:5]),self.shell.predict(X[:,:5])]);r=np.exp(X[:,5])
        shift=self.tau*(p[:,1]-r*p[:,0])/(r*r*self.variances[0]+self.variances[1])
        result=p-np.column_stack([-r*self.variances[0]*shift,self.variances[1]*shift])
        if not np.isfinite(result).all() or (result<=0).any():
            raise ValueError('Nonpositive or nonfinite area: candidate needs physical-domain review')
        return result
