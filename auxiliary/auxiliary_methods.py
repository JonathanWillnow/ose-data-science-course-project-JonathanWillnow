"""This module contains auxiliary functions for all different kinds of methods which are used in the main notebook."""
import numpy as np
import pandas as pd
import pandas.io.formats.style
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm_api
import geopandas 
import matplotlib.pyplot as plt
import seaborn as sns
import linearmodels as lm
import itertools as tools
import scipy.stats as scipy
from IPython.display import HTML

from auxiliary.auxiliary_regressions import *


def OveridentifyingTest_after_Sargan(FS, SS, data, plot):
    
    """ Function to calculate the Chisquared p value for the overidentifying Test after Sargan for the case when we 
    calculated the 2SLS estimator by performing first- and second-stage regression seperatly. The function was designed
    after Prof. Dr. Horst Rottmann (https://wirtschaftslexikon.gabler.de/definition/sargan-test-52105) and by taking 
    the intuition of Ben Lambert, a research associate at the Imperial College London.
    
    https://wirtschaftslexikon.gabler.de/definition/sargan-test-52105
    https://ben-lambert.com/about/

        Args:
        -------
            FS = first-stage regression model
            SS = second-stage regression model
            plot = boolean, if True we get a plot 

        Returns:
        ---------
            p_value_chi
            R_squared of regressing the residuals on the exogenous variables
    """
  
    eps = SS.resids
    variables = FS.model.exog.vars[1:]
    dependent = eps
    exog = data[variables]
    mod = lm.panel.PanelOLS(dependent, exog, time_effects = True, entity_effects=True).fit()
    
    r_squared = mod.rsquared
    x = np.arange(0, 10, 0.001)
    point = r_squared * SS.resids.shape[0]

    p_value_chi = 1 - scipy.chi2.cdf(point, 1)
    if plot == True:
        f, axs = plt.subplots()

        ax = plt.plot(x, scipy.chi2.pdf(x, df=1))
        ax = plt.axvline(x=point, color= "r", label = "SARG")
        plt.legend()
    
    return(round(p_value_chi,4), round(r_squared,4))

###


def all_p_values(data):
    
    OLS1, FS1, SS1, OLS2, FS2, SS2 = OFn_OFa_all_Table2(data, table = 0)
    OLS1_oofv, FS1_oofv, SS1_oofv, OLS2_oofv, FS2_oofv, SS2_oofv = OFn_OFa_oofv_Table2(data, table = 0)
    OLS1_oda, FS1_oda, SS1_oda, OLS2_oda, FS2_oda, SS2_oda = OFn_OFa_oda_Table2(data, table = 0)
    
    result = pd.DataFrame(columns = ["p_value", "R_squared"], index=["OFn_all", "OFa_all_ln", "OFn_oofv",
                                                        "OFa_oofv_ln", "OFn_oda", "OFa_oda_ln"])
    for a,b,c in zip([SS1,SS2,SS1_oofv, SS2_oofv, SS1_oda, SS2_oda], [FS1, FS2, FS1_oofv, FS2_oofv, FS1_oda, FS2_oda], result.index):
        pval, rsquared = OveridentifyingTest_after_Sargan(b, a, data, False)
        result.loc[c,"p_value"] = pval
        result.loc[c,"R_squared"] = rsquared
            
    return(result)
                          
    
    