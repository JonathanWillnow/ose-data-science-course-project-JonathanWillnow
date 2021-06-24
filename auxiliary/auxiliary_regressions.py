

import numpy as np
import pandas as pd
import pandas.io.formats.style
import seaborn as sns
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm_api
import matplotlib.pyplot as plt
import geopandas 
import linearmodels as lm
import itertools as tools
from IPython.display import HTML


def OFn_OFa_all_Table2(data4, plotting):
    # This was constructed by trying to copy the provided stata code
    # Stata
    #
    # OOF projects count and amount

    dependent = data4.growth_pc
    exog = sm.tools.add_constant(data4[["l2OFn_all", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependent, exog, entity_effects = True, time_effects = True)
    mod_OLS1 = mod.fit(cov_type='unadjusted')
    #print(mod_OLS1.conf_int(level = 0.9).loc["l2OFn_all", "lower"])
    
    # same using cov_type = "clustered" does NOT change anything.

    dependendFS = data4.l2OFn_all
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probabilityOFn_all","l3factor1*probabilityOFn_all"]])
    mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
    mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

    fitted_c = mod_FS1.fitted_values
    data4["Chinese_OFn_(t-2)"] = fitted_c

    dependentSS = data4.growth_pc
    exog = sm.tools.add_constant(data4[["Chinese_OFn_(t-2)", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
    mod_SS1 = mod.fit(cov_type='clustered', clusters = data4.code)

    # All projects (log) financial amounts

    dependent = data4.growth_pc
    exog = sm.tools.add_constant(data4[["l2OFa_all", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependent, exog, entity_effects = True, time_effects = True)
    mod_OLS2 = mod.fit(cov_type='unadjusted')

    # same using cov_type = "clustered" does NOT change anything.

    dependendFS = data4.l2OFa_all
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probabilityOFa_all","l3factor1*probabilityOFa_all"]])
    mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
    mod_FS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    fitted_c = mod_FS2.fitted_values
    data4["Chinese_OFa_(t-2)"] = fitted_c

    dependentSS = data4.growth_pc
    exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
    mod_SS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    if plotting == True:
        return(print(lm.panel.compare({"OLS OFn_all": mod_OLS1,"FS OFn_all": mod_FS1, "SS OFn_all": mod_SS1,
                               "OLS OFa_all": mod_OLS2,"FS OFa_all": mod_FS2, "SS OFa_all": mod_SS2},
                               stars = True, precision = "std_errors")))
    else:
        return(mod_OLS1, mod_FS1, mod_SS1, mod_OLS2, mod_FS2, mod_SS2)
    
    
    ###########################################
    
    
def OFn_OFa_oda_Table2(data4, plotting):
    # This was constructed by trying to copy the provided stata code
    # Stata
    #
    # OOF projects count and amount

    dependent = data4.growth_pc
    exog = sm.tools.add_constant(data4[["l2OFn_oda", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependent, exog, entity_effects = True, time_effects = True)
    mod_OLS1 = mod.fit(cov_type='unadjusted')
    #print(mod_OLS1.conf_int(level = 0.9).loc["l2OFn_all", "lower"])
    
    # same using cov_type = "clustered" does NOT change anything.

    dependendFS = data4.l2OFn_oda
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probabilityOFn_oda","l3factor1*probabilityOFn_oda"]])
    mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
    mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

    fitted_c = mod_FS1.fitted_values
    data4["Chinese_OFn_(t-2)"] = fitted_c

    dependentSS = data4.growth_pc
    exog = sm.tools.add_constant(data4[["Chinese_OFn_(t-2)", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
    mod_SS1 = mod.fit(cov_type='clustered', clusters = data4.code)

    # All projects (log) financial amounts

    dependent = data4.growth_pc
    exog = sm.tools.add_constant(data4[["l2OFa_oda", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependent, exog, entity_effects = True, time_effects = True)
    mod_OLS2 = mod.fit(cov_type='unadjusted')

    # same using cov_type = "clustered" does NOT change anything.

    dependendFS = data4.l2OFa_oda
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probabilityOFa_oda","l3factor1*probabilityOFa_oda"]])
    mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
    mod_FS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    fitted_c = mod_FS2.fitted_values
    data4["Chinese_OFa_(t-2)"] = fitted_c

    dependentSS = data4.growth_pc
    exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
    mod_SS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    if plotting == True:
        return(print(lm.panel.compare({"OLS OFn_oda": mod_OLS1,"FS OFn_oda": mod_FS1, "SS OFn_oda": mod_SS1,
                               "OLS OFa_oda": mod_OLS2,"FS OFa_oda": mod_FS2, "SS OFa_oda": mod_SS2},
                               stars = True, precision = "std_errors")))
    else:
        return(mod_OLS1, mod_FS1, mod_SS1, mod_OLS2, mod_FS2, mod_SS2)
    
    
    ####################################################
    
def OFn_OFa_oofv_Table2(data4, plotting):
    # This was constructed by trying to copy the provided stata code
    # Stata
    #
    # OOF projects count and amount

    dependent = data4.growth_pc
    exog = sm.tools.add_constant(data4[["l2OFn_oofv", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependent, exog, entity_effects = True, time_effects = True)
    mod_OLS1 = mod.fit(cov_type='unadjusted')
    #print(mod_OLS1.conf_int(level = 0.9).loc["l2OFn_all", "lower"])
    
    # same using cov_type = "clustered" does NOT change anything.

    dependendFS = data4.l2OFn_oofv
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probabilityOFn_oofv","l3factor1*probabilityOFn_oofv"]])
    mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
    mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

    fitted_c = mod_FS1.fitted_values
    data4["Chinese_OFn_(t-2)"] = fitted_c

    dependentSS = data4.growth_pc
    exog = sm.tools.add_constant(data4[["Chinese_OFn_(t-2)", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
    mod_SS1 = mod.fit(cov_type='clustered', clusters = data4.code)

    # All projects (log) financial amounts

    dependent = data4.growth_pc
    exog = sm.tools.add_constant(data4[["l2OFa_oofv", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependent, exog, entity_effects = True, time_effects = True)
    mod_OLS2 = mod.fit(cov_type='unadjusted')

    # same using cov_type = "clustered" does NOT change anything.

    dependendFS = data4.l2OFa_oofv
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probabilityOFa_oofv","l3factor1*probabilityOFa_oofv"]])
    mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
    mod_FS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    fitted_c = mod_FS2.fitted_values
    data4["Chinese_OFa_(t-2)"] = fitted_c

    dependentSS = data4.growth_pc
    exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
    mod_SS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    if plotting == True:
        return(print(lm.panel.compare({"OLS OFn_oofv": mod_OLS1,"FS OFn_oofv": mod_FS1, "SS OFn_oofv": mod_SS1,
                               "OLS OFa_oofv": mod_OLS2,"FS OFa_oofv": mod_FS2, "SS OFa_oofv": mod_SS2},
                               stars = True, precision = "std_errors")))
    else:
        return(mod_OLS1, mod_FS1, mod_SS1, mod_OLS2, mod_FS2, mod_SS2)
    
    
    #########################################
    
def confidence_intervall_plot(data, alpha):
    
    """ Function to plot all the confidence intervalls of the regressions that are used to replicate table 2

        Args:
        -------
            data = pd.dataframe wich contains data to do the regressions
            alpha = significance level

        Returns:
        ---------
            plot
    """
    
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 100
    
    # set up dic and initialise variables
    data_dict = {}
    data_dict['variable'] = [data.l2OFn_all,  data.l2OFa_all, data.l2OFn_oda,  data.l2OFa_oda, data.l2OFn_oofv,  data.l2OFa_oofv]
    
    # get all the models
    ao1, af1, as1, ao2, af2, as3 = OFn_OFa_all_Table2(data, False)
    oda_o1, oda_f1, oda_s1, oda_o2, oda_f2, oda_s3 = OFn_OFa_oda_Table2(data, False)
    oofv_o1, oofv_f1, oofv_s1, oofv_o2, oofv_f2, oofv_s3 = OFn_OFa_oofv_Table2(data, False)
    
    # calculate 90% CI
    data_dict['low'] = [ao1.conf_int(level = 1-alpha).loc["l2OFn_all", "lower"], 
                        ao2.conf_int(level = 1-alpha).loc["l2OFa_all", "lower"],
                       oda_o1.conf_int(level = 1-alpha).loc["l2OFn_oda", "lower"], 
                        oda_o2.conf_int(level = 1-alpha).loc["l2OFa_oda", "lower"],
                       oofv_o1.conf_int(level = 1-alpha).loc["l2OFn_oofv", "lower"], 
                        oofv_o2.conf_int(level = 1-alpha).loc["l2OFa_oofv", "lower"]]
    data_dict['up'] = [ao1.conf_int(level = 1-alpha).loc["l2OFn_all", "upper"], 
                       ao2.conf_int(level = 1-alpha).loc["l2OFa_all", "upper"],
                      oda_o1.conf_int(level = 1-alpha).loc["l2OFn_oda", "upper"], 
                       oda_o2.conf_int(level = 1-alpha).loc["l2OFa_oda", "upper"],
                       oofv_o1.conf_int(level = 1-alpha).loc["l2OFn_oofv", "upper"], 
                       oofv_o2.conf_int(level = 1-alpha).loc["l2OFa_oofv", "upper"]]
    
    dataset = pd.DataFrame(data_dict)
    
    # set up style and plot in loop
    #fig.suptitle('Comparison of 90% CI´s', fontsize=25)
    #plt.pyplot.figure(figsize=(20,14))
    col = ['ro-','ro-', "go-","go-","bo-","bo-"]
    labels = ["OFn_all", "OFa_all", "OFn_oda", "OFa_oda", "OFn_oofv", "OFa_oofv"]
    x = 0
    for low,up,y in zip(dataset['low'],dataset['up'],range(len(dataset))):
        plt.plot((low,up),(y,y),col[x], label = labels[x])  
        x +=1
    plt.yticks(list(range(len(dataset))), ["OFn_all", "OFa_all", "OFn_oda", "OFa_oda", "OFn_oofv", "OFa_oofv"])
    #plt.pyplot.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    
    