

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


def OFn_OFa_all_Table2(data4, table):
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
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFn_all","l3factor1*probOFn_all"]])
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
    exog = sm.tools.add_constant(data4[["l2OFa_all_ln", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependent, exog, entity_effects = True, time_effects = True)
    mod_OLS2 = mod.fit(cov_type='unadjusted')

    # same using cov_type = "clustered" does NOT change anything.

    dependendFS = data4.l2OFa_all_ln
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFa_all_ln","l3factor1*probOFa_all_ln"]])
    mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
    mod_FS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    fitted_c = mod_FS2.fitted_values
    data4["Chinese_OFa_(t-2)"] = fitted_c

    dependentSS = data4.growth_pc
    exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
    mod_SS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    if table == True:
        return(print(lm.panel.compare({"OLS OFn_all": mod_OLS1,"FS OFn_all": mod_FS1, "SS OFn_all": mod_SS1,
                               "OLS OFa_all": mod_OLS2,"FS OFa_all": mod_FS2, "SS OFa_all": mod_SS2},
                               stars = True, precision = "std_errors")))
    else:
        return(mod_OLS1, mod_FS1, mod_SS1, mod_OLS2, mod_FS2, mod_SS2)
    
    
    ###########################################
    
    
def OFn_OFa_oda_Table2(data4, table):
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
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFn_oda","l3factor1*probOFn_oda"]])
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
    exog = sm.tools.add_constant(data4[["l2OFa_oda_ln", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependent, exog, entity_effects = True, time_effects = True)
    mod_OLS2 = mod.fit(cov_type='unadjusted')

    # same using cov_type = "clustered" does NOT change anything.

    dependendFS = data4.l2OFa_oda_ln
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFa_oda_ln","l3factor1*probOFa_oda_ln"]])
    mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
    mod_FS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    fitted_c = mod_FS2.fitted_values
    data4["Chinese_OFa_(t-2)"] = fitted_c

    dependentSS = data4.growth_pc
    exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
    mod_SS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    if table == True:
        return(print(lm.panel.compare({"OLS OFn_oda": mod_OLS1,"FS OFn_oda": mod_FS1, "SS OFn_oda": mod_SS1,
                               "OLS OFa_oda": mod_OLS2,"FS OFa_oda": mod_FS2, "SS OFa_oda": mod_SS2},
                               stars = True, precision = "std_errors")))
    else:
        return(mod_OLS1, mod_FS1, mod_SS1, mod_OLS2, mod_FS2, mod_SS2)
    
    
    ####################################################
    
def OFn_OFa_oofv_Table2(data4, table):
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
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFn_oofv","l3factor1*probOFn_oofv"]])
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
    exog = sm.tools.add_constant(data4[["l2OFa_oofv_ln", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependent, exog, entity_effects = True, time_effects = True)
    mod_OLS2 = mod.fit(cov_type='unadjusted')

    # same using cov_type = "clustered" does NOT change anything.

    dependendFS = data4.l2OFa_oofv_ln
    exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFa_oofv_ln","l3factor1*probOFa_oofv_ln"]])
    mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
    mod_FS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    fitted_c = mod_FS2.fitted_values
    data4["Chinese_OFa_(t-2)"] = fitted_c

    dependentSS = data4.growth_pc
    exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln"]])
    mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
    mod_SS2 = mod.fit(cov_type='clustered', clusters = data4.code)

    if table == True:
        return(print(lm.panel.compare({"OLS OFn_oofv": mod_OLS1,"FS OFn_oofv": mod_FS1, "SS OFn_oofv": mod_SS1,
                               "OLS OFa_oofv": mod_OLS2,"FS OFa_oofv": mod_FS2, "SS OFa_oofv": mod_SS2},
                               stars = True, precision = "std_errors")))
    else:
        return(mod_OLS1, mod_FS1, mod_SS1, mod_OLS2, mod_FS2, mod_SS2)
    
    
    #########################################
    
def confidence_intervall_plot(data, alpha, exog_var):
    
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
    
    # set up dic and initialise variables
    data_dict = {}
    data_dict['variable'] = [data.l2OFn_all,  data.l2OFa_all_ln, data.l2OFn_oofv,  data.l2OFa_oofv_ln, data.l2OFn_oda,  data.l2OFa_oda_ln]
    
    # get all the models
    if exog_var == ("l2Exports_ln" or "l2FDI_China_ln"):
        as1,oofv_s1, oda_s1, as2, oofv_s2, oda_s2 = OFn_OFa_all_Table2_robustness(data, False, [exog_var])
        
        data_dict = {}
        data_dict['variable'] = [data.l2OFn_all,  data.l2OFa_all_ln, data.l2OFn_oofv,  data.l2OFa_oofv_ln,
                                  data.l2OFn_oda,  data.l2OFa_oda_ln]
        # calculate 90% CI
        data_dict['low'] = [as1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "lower"], 
                                as2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "lower"],
                               oofv_s1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "lower"], 
                                oofv_s2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "lower"], 
                               oda_s1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "lower"], 
                                oda_s2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "lower"]]
        data_dict['up'] = [as1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "upper"], 
                               as2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "upper"],
                               oofv_s1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "upper"], 
                               oofv_s2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "upper"],
                              oda_s1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "upper"], 
                               oda_s2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "upper"]]

        dataset_robust = pd.DataFrame(data_dict)

        col = ['ro-','ro-', "go-","go-","bo-","bo-"]
        labels = ["OFn_all", "OFa_all_ln", "OFn_oofv", "OFa_oofv_ln", "OFn_oda", "OFa_oda_ln"]

        fig, ax = plt.subplots(figsize=(14,6)) 
        x = 0
        for low,up,y in zip(dataset_robust['low'],dataset_robust['up'],range(len(dataset_robust))):
            ax = plt.plot((low,up),(y,y),col[x], label = labels[x])  
            x +=1
        plt.yticks(list(range(len(dataset_robust))), ["OFn_all", "OFa_all_ln", "OFn_oofv", "OFa_oofv_ln", "OFn_oda", "OFa_oda_ln"])
        plt.xlabel("Robustness tests: Effect on Growth for choosen specification")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.tight_layout(pad=2.5)
        
 
        ###
        
    else:
        ao1, af1, as1, ao2, af2, as2 = OFn_OFa_all_Table2(data, False)
        oda_o1, oda_f1, oda_s1, oda_o2, oda_f2, oda_s2 = OFn_OFa_oda_Table2(data, False)
        oofv_o1, oofv_f1, oofv_s1, oofv_o2, oofv_f2, oofv_s2 = OFn_OFa_oofv_Table2(data, False)
    
        
    
        # calculate 90% CI
        data_dict['low'] = [ao1.conf_int(level = 1-alpha).loc["l2OFn_all", "lower"], 
                                ao2.conf_int(level = 1-alpha).loc["l2OFa_all_ln", "lower"],
                               oofv_o1.conf_int(level = 1-alpha).loc["l2OFn_oofv", "lower"], 
                                oofv_o2.conf_int(level = 1-alpha).loc["l2OFa_oofv_ln", "lower"], 
                               oda_o1.conf_int(level = 1-alpha).loc["l2OFn_oda", "lower"], 
                                oda_o2.conf_int(level = 1-alpha).loc["l2OFa_oda_ln", "lower"]]
        data_dict['up'] = [ao1.conf_int(level = 1-alpha).loc["l2OFn_all", "upper"], 
                               ao2.conf_int(level = 1-alpha).loc["l2OFa_all_ln", "upper"],
                               oofv_o1.conf_int(level = 1-alpha).loc["l2OFn_oofv", "upper"], 
                               oofv_o2.conf_int(level = 1-alpha).loc["l2OFa_oofv_ln", "upper"],
                              oda_o1.conf_int(level = 1-alpha).loc["l2OFn_oda", "upper"], 
                               oda_o2.conf_int(level = 1-alpha).loc["l2OFa_oda_ln", "upper"]]

        dataset = pd.DataFrame(data_dict)


        data_dict2 = {}
        data_dict2['variable'] = [data.l2OFn_all,  data.l2OFa_all_ln, data.l2OFn_oofv,  data.l2OFa_oofv_ln,
                                  data.l2OFn_oda,  data.l2OFa_oda_ln]
        # calculate 90% CI
        data_dict2['low'] = [as1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "lower"], 
                                as2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "lower"],
                               oofv_s1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "lower"], 
                                oofv_s2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "lower"], 
                               oda_s1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "lower"], 
                                oda_s2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "lower"]]
        data_dict2['up'] = [as1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "upper"], 
                               as2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "upper"],
                               oofv_s1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "upper"], 
                               oofv_s2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "upper"],
                              oda_s1.conf_int(level = 1-alpha).loc["Chinese_OFn_(t-2)", "upper"], 
                               oda_s2.conf_int(level = 1-alpha).loc["Chinese_OFa_(t-2)", "upper"]]

        dataset2 = pd.DataFrame(data_dict2)

        col = ['ro-','ro-', "go-","go-","bo-","bo-"]
        labels = ["OFn_all", "OFa_all_ln", "OFn_oofv", "OFa_oofv_ln", "OFn_oda", "OFa_oda_ln"]

        fig, ax = plt.subplots(1,2, figsize=(14,6)) 
        x = 0
        plt.subplot(121)
        for low,up,y in zip(dataset['low'],dataset['up'],range(len(dataset))):
            ax = plt.plot((low,up),(y,y),col[x], label = labels[x])  
            x +=1
        plt.yticks(list(range(len(dataset))), ["OFn_all", "OFa_all_ln", "OFn_oofv", "OFa_oofv_ln", "OFn_oda", "OFa_oda_ln"])
        plt.xlabel("Effect on Growth for OLS")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        x = 0
        plt.subplot(122)
        for low,up,y in zip(dataset2['low'],dataset2['up'],range(len(dataset2))):
            ax = plt.plot((low,up),(y,y),col[x], label = labels[x])  
            x +=1
        plt.yticks(list(range(len(dataset))), ["OFn_all", "OFa_all_ln", "OFn_oofv", "OFa_oofv_ln", "OFn_oda", "OFa_oda_ln"])
        plt.xlabel("Effect on Growth for 2SLS")


        plt.tight_layout(pad=2.5)
    
    
    
    
#################################################


def OFn_OFa_all_Table2_robustness(data4, table, exog_var):
   
    """ Function calculate the gowth effects of Chinese development finance, 2SLS, with additional controls

        Args:
        -------
            data = pd.dataframe wich contains data to do the regressions
            plotting = bolean, if True this will return a linearmodels comparison table
            exog_var = single string or list of strings containing the additional variables for that we 
                       want to controll

        Returns:
        ---------
            if table = True a regression table, if False all regressions seperatly
    """

    if len(exog_var) == 1:
        exog_var = exog_var[0]
        dependendFS = data4.l2OFn_all
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFn_all","l3factor1*probOFn_all", exog_var]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFn_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFn_(t-2)", "l1population_ln", exog_var]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS1 = mod.fit(cov_type='clustered', clusters = data4.code)
        ########################
        dependendFS = data4.l2OFn_oofv
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFn_oofv","l3factor1*probOFn_oofv", exog_var]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFn_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFn_(t-2)", "l1population_ln", exog_var]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS2 = mod.fit(cov_type='clustered', clusters = data4.code)
        ########################
        dependendFS = data4.l2OFn_oda
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFn_oda","l3factor1*probOFn_oda", exog_var]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFn_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFn_(t-2)", "l1population_ln", exog_var]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS3 = mod.fit(cov_type='clustered', clusters = data4.code)
        #
        # amounst
        #
        dependendFS = data4.l2OFa_all_ln
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFa_all_ln","l3factor1*probOFa_all_ln", exog_var]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFa_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln", exog_var]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS1a = mod.fit(cov_type='clustered', clusters = data4.code)
        ########################
        dependendFS = data4.l2OFa_oofv_ln
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFa_oofv_ln","l3factor1*probOFa_oofv_ln",exog_var]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFa_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln", exog_var]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS2a = mod.fit(cov_type='clustered', clusters = data4.code)
        ########################
        dependendFS = data4.l2OFa_oda_ln
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFa_oda_ln","l3factor1*probOFa_oda_ln", exog_var]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFa_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln", exog_var]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS3a = mod.fit(cov_type='clustered', clusters = data4.code)



        if table == True:
            return(print(lm.panel.compare({"SS OFn_all": mod_SS1, "SS OFn_oofv" : mod_SS2, "SS OFn_oda" : mod_SS3,
                                          "SS OFa_all": mod_SS1a, "SS OFa_oofv" : mod_SS2a, "SS OFa_oda" : mod_SS3a},
                                   stars = True, precision = "std_errors")))
        else:
            return(mod_SS1,mod_SS2, mod_SS3, mod_SS1a, mod_SS2a, mod_SS3a)
    
#############################################################################################    
    
    elif len(exog_var) == 2:
        
        exog_var1 = exog_var[0]
        exog_var2 = exog_var[1]
        
        dependendFS = data4.l2OFn_all
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFn_all","l3factor1*probOFn_all", exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFn_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFn_(t-2)", "l1population_ln", exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS1 = mod.fit(cov_type='clustered', clusters = data4.code)
        ########################
        dependendFS = data4.l2OFn_oofv
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFn_oofv","l3factor1*probOFn_oofv", exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFn_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFn_(t-2)", "l1population_ln", exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS2 = mod.fit(cov_type='clustered', clusters = data4.code)
        ########################
        dependendFS = data4.l2OFn_oda
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFn_oda","l3factor1*probOFn_oda", exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFn_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFn_(t-2)", "l1population_ln", exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS3 = mod.fit(cov_type='clustered', clusters = data4.code)
        #
        # amounst
        #
        dependendFS = data4.l2OFa_all_ln
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFa_all_ln","l3factor1*probOFa_all_ln", 
                                             exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFa_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln", exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS1a = mod.fit(cov_type='clustered', clusters = data4.code)
        ########################
        dependendFS = data4.l2OFa_oofv_ln
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFa_oofv_ln","l3factor1*probOFa_oofv_ln",
                                             exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFa_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln", exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS2a = mod.fit(cov_type='clustered', clusters = data4.code)
        ########################
        dependendFS = data4.l2OFa_oda_ln
        exog2 = sm.tools.add_constant(data4[["l1population_ln", "l3Reserves*probOFa_oda_ln","l3factor1*probOFa_oda_ln",
                                             exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True)
        mod_FS1 = mod.fit(cov_type='clustered', clusters = data4.code)

        fitted_c = mod_FS1.fitted_values
        data4["Chinese_OFa_(t-2)"] = fitted_c

        dependentSS = data4.growth_pc
        exog = sm.tools.add_constant(data4[["Chinese_OFa_(t-2)", "l1population_ln", exog_var1, exog_var2]])
        mod = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True)
        mod_SS3a = mod.fit(cov_type='clustered', clusters = data4.code)



        if table == True:
            return(print(lm.panel.compare({"SS OFn_all": mod_SS1, "SS OFn_oofv" : mod_SS2, "SS OFn_oda" : mod_SS3,
                                          "SS OFa_all": mod_SS1a, "SS OFa_oofv" : mod_SS2a, "SS OFa_oda" : mod_SS3a},
                                   stars = True, precision = "std_errors")))
        else:
            return(mod_SS1,mod_SS2, mod_SS3, mod_SS1a, mod_SS2a, mod_SS3a)
    else:
        print("Wrong input")

    