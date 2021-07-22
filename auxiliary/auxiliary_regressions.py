"""This module contains all the regressions which are used in the main notebook."""
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
import warnings 
from IPython.display import HTML


def replicating_table1(allocation_data_lagg):
    # rebuilding whole table 1
    # prob_PRCaid_2000" = OF probability historic
    exog_variables = ["A_LINLINECHN","A_Ltaiwanr","A_Ltrade_con_ln","A_D99petroleum",
                      "A_LDebtGDP","A_Lpolity2","A_Lgdppc_con_ln", "A_Lpopulation_ln","A_Lenglish"]
    exog = sm.tools.add_constant(allocation_data_lagg[exog_variables])
    #1
    mod = lm.panel.RandomEffects(allocation_data_lagg.OFn_all, exog)
    mod_random1 = mod.fit(cov_type='clustered', clusters = allocation_data_lagg.code)
    #2
    exog_variables = ["factor1_1","A_LINLINECHN","A_Ltaiwanr","A_Ltrade_con_ln","A_D99petroleum",
                      "A_LDebtGDP","A_Lpolity2","A_Lgdppc_con_ln", "A_Lpopulation_ln","A_Lenglish"]
    exog = sm.tools.add_constant(allocation_data_lagg[exog_variables])
    mod = lm.panel.RandomEffects(allocation_data_lagg.OFn_all, exog)
    mod_random2 = mod.fit(cov_type='clustered', clusters = allocation_data_lagg.code)

    #3
    exog_variables = ["det_reservesCHN_con_1","A_LINLINECHN","A_Ltaiwanr","A_Ltrade_con_ln","A_D99petroleum",
                      "A_LDebtGDP","A_Lpolity2","A_Lgdppc_con_ln", "A_Lpopulation_ln","A_Lenglish"]
    exog = sm.tools.add_constant(allocation_data_lagg[exog_variables])
    mod = lm.panel.RandomEffects(allocation_data_lagg.OFn_all, exog)
    mod_random3 = mod.fit(cov_type='clustered', clusters = allocation_data_lagg.code)

    #4 historic probability?
    exog_variables = ["factor1_1","prob_PRCaid_2000","A_LINLINECHN","A_Ltaiwanr","A_Ltrade_con_ln","A_D99petroleum",
                      "A_LDebtGDP","A_Lpolity2","A_Lgdppc_con_ln", "A_Lpopulation_ln","A_Lenglish"]
    exog = sm.tools.add_constant(allocation_data_lagg[exog_variables])
    mod = lm.panel.RandomEffects(allocation_data_lagg.OFn_all, exog)
    mod_random4 = mod.fit(cov_type='clustered', clusters = allocation_data_lagg.code)

    #5 contempt probability?
    exog_variables = ["factor1_1","probaid_PRC_OFn_all","A_LINLINECHN","A_Ltaiwanr","A_Ltrade_con_ln","A_D99petroleum",
                      "A_LDebtGDP","A_Lpolity2","A_Lgdppc_con_ln", "A_Lpopulation_ln","A_Lenglish"]
    exog = sm.tools.add_constant(allocation_data_lagg[exog_variables])
    mod = lm.panel.RandomEffects(allocation_data_lagg.OFn_all, exog)
    mod_random5 = mod.fit(cov_type='clustered', clusters = allocation_data_lagg.code)

    #6
    exog_variables = ["A_LINLINECHN","A_Ltaiwanr","A_Ltrade_con_ln","A_D99petroleum",
                      "A_LDebtGDP","A_Lpolity2","A_Lgdppc_con_ln", "A_Lpopulation_ln","A_Lenglish", "factor1_1"]
    exog = sm.tools.add_constant(allocation_data_lagg[exog_variables])
    mod = lm.panel.PanelOLS(allocation_data_lagg.OFn_all, exog, entity_effects=True, drop_absorbed=True)
    mod_fixed1 = mod.fit(cov_type='clustered', clusters = allocation_data_lagg.code);

    # 7
    exog_variables = ["det_reservesCHN_con_1","A_LINLINECHN","A_Ltaiwanr","A_Ltrade_con_ln","A_D99petroleum",
                      "A_LDebtGDP","A_Lpolity2","A_Lgdppc_con_ln", "A_Lpopulation_ln","A_Lenglish"]
    exog = sm.tools.add_constant(allocation_data_lagg[exog_variables])
    mod = lm.panel.PanelOLS(allocation_data_lagg.OFn_all, exog, entity_effects= True, drop_absorbed= True)
    mod_fixed2 = mod.fit(cov_type='clustered', clusters = allocation_data_lagg.code);


    print(lm.panel.compare({"RE1": mod_random1, "RE2": mod_random2, "RE3": mod_random3, "RE4" : mod_random4, "RE5":mod_random5},
                  stars = True, precision = "std_errors"))
    print(lm.panel.compare({"FE1":mod_fixed1, "FE2":mod_fixed2},
                  stars = True, precision = "std_errors"))




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
    if exog_var == "l2Exports_ln" or exog_var == "l2FDI_China_ln":
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

        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

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
    warnings.filterwarnings("ignore")
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

    
    
#####################################################################################################
    
#####################################################################################################
    
def various_lags_2SLS_table(data_lagged):

    table3_results = pd.DataFrame(index=['Chinese OF (t+1)', 'SE (t+1)', 'Chinese OF (t+0)','SE (t+0)','Chinese OF (t-1)',
                                         'SE (t-1)','Chinese OF (t-2)','SE (t-2)','Chinese OF (t-3)','SE (t-3)','Chinese OF (t-4)',
                                         'SE (t-4)', 'Chinese OF (t-5)','SE (t-5)','Chinese OF (t-6)', 'SE (t-6)'])
    
    warnings.filterwarnings("ignore")
    ################################ Project count ###################################
    ################################# All projects ###################################
    ## t+1 all
    dependendFS = data_lagged.f1OFn_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "Reserves*probOFn_all","factor1*probOFn_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn(t+1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn(t+1)", "l1population_ln"]])
    mod_tplus1_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tplus1_all = round(mod_tplus1_all.params[1],3)
    ## t all
    dependendFS = data_lagged.OFn_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l1Reserves*probOFn_all","l1factor1*probOFn_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn(t)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn(t)", "l1population_ln"]])
    mod_t_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    t_all = round(mod_t_all.params[1],3)
    ## t -1 all
    dependendFS = data_lagged.l1OFn_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l2Reserves*probOFn_all","l2factor1*probOFn_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn(t-1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn(t-1)", "l1population_ln"]])
    mod_tminus1_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus1_all = round(mod_tminus1_all.params[1],3)
    sm1 = round(mod_tminus1_all.std_errors[1],3)
    ## t -2 all
    dependendFS = data_lagged.l2OFn_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l3Reserves*probOFn_all","l3factor1*probOFn_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn(t-2)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_tminus2_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus2_all = round(mod_tminus2_all.params[1],3)
    ## t -3 all
    dependendFS = data_lagged.l3OFn_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l4Reserves*probOFn_all","l4factor1*probOFn_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn(t-3)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn(t-3)", "l1population_ln"]])
    mod_tminus3_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus3_all = round(mod_tminus3_all.params[1],3)
   
    ## t -4 all
    dependendFS = data_lagged.l4OFn_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l5Reserves*probOFn_all","l5factor1*probOFn_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn(t-4)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn(t-4)", "l1population_ln"]])
    mod_tminus4_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus4_all = round(mod_tminus4_all.params[1],3)
    ## t -5 all
    dependendFS = data_lagged.l5OFn_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l6Reserves*probOFn_all","l6factor1*probOFn_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn(t-5)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn(t-5)", "l1population_ln"]])
    mod_tminus5_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus5_all = round(mod_tminus5_all.params[1],3)
   
    ## t -6 all
    dependendFS = data_lagged.l6OFn_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l7Reserves*probOFn_all","l7factor1*probOFn_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn(t-6)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn(t-6)", "l1population_ln"]])
    mod_tminus6_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus6_all = round(mod_tminus6_all.params[1],3)
    
    sp1 = round(mod_tplus1_all.std_errors[1],3)
    s = round(mod_t_all.std_errors[1],3)
    sm2 = round(mod_tminus2_all.std_errors[1],3)
    sm3 = round(mod_tminus3_all.std_errors[1],3)
    sm4 = round(mod_tminus4_all.std_errors[1],3)
    sm5 = round(mod_tminus5_all.std_errors[1],3)
    sm6 = round(mod_tminus1_all.std_errors[1],3)

    ####################################### results ###########################################
    table3_results["All projects"] = [tplus1_all,f"({sp1})", t_all, f"({s})", tminus1_all, f"({sm1})", tminus2_all, f"({sm2})", tminus3_all, f"({sm3})",
                                      tminus4_all,f"({sm4})", tminus5_all, f"({sm5})", tminus6_all, f"({sm6})"]
    
    ################################# OOF projects ###################################
    ## t+1 all
    dependendFS = data_lagged.f1OFn_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "Reserves*probOFn_oofv","factor1*probOFn_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oofv(t+1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oofv(t+1)", "l1population_ln"]])
    mod_tplus1_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tplus1_oofv = round(mod_tplus1_oofv.params[1],3)
    ## t all
    dependendFS = data_lagged.OFn_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l1Reserves*probOFn_oofv","l1factor1*probOFn_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oofv(t)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oofv(t)", "l1population_ln"]])
    mod_t_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    t_oofv = round(mod_t_oofv.params[1],3)
    ## t -1 all
    dependendFS = data_lagged.l1OFn_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l2Reserves*probOFn_oofv","l2factor1*probOFn_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oofv(t-1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oofv(t-1)", "l1population_ln"]])
    mod_tminus1_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus1_oofv = round(mod_tminus1_oofv.params[1],3)
    ## t -2 all
    dependendFS = data_lagged.l2OFn_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l3Reserves*probOFn_oofv","l3factor1*probOFn_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oofv(t-2)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oofv(t-2)", "l1population_ln"]])
    mod_tminus2_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus2_oofv = round(mod_tminus2_oofv.params[1],3)
    ## t -3 all
    dependendFS = data_lagged.l3OFn_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l4Reserves*probOFn_oofv","l4factor1*probOFn_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oofv(t-3)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oofv(t-3)", "l1population_ln"]])
    mod_tminus3_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus3_oofv = round(mod_tminus3_oofv.params[1],3)
    ## t -4 all
    dependendFS = data_lagged.l4OFn_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l5Reserves*probOFn_oofv","l5factor1*probOFn_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oofv(t-4)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oofv(t-4)", "l1population_ln"]])
    mod_tminus4_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus4_oofv = round(mod_tminus4_oofv.params[1],3)
    ## t -5 all
    dependendFS = data_lagged.l5OFn_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l6Reserves*probOFn_oofv","l6factor1*probOFn_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oofv(t-5)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oofv(t-5)", "l1population_ln"]])
    mod_tminus5_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus5_oofv = round(mod_tminus5_oofv.params[1],3)
    ## t -6 all
    dependendFS = data_lagged.l6OFn_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l7Reserves*probOFn_oofv","l7factor1*probOFn_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oofv(t-6)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oofv(t-6)", "l1population_ln"]])
    mod_tminus6_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus6_oofv = round(mod_tminus6_oofv.params[1],3)
    
    sp1 = round(mod_tplus1_oofv.std_errors[1],3)
    s = round(mod_t_oofv.std_errors[1],3)
    sm2 = round(mod_tminus2_oofv.std_errors[1],3)
    sm3 = round(mod_tminus3_oofv.std_errors[1],3)
    sm4 = round(mod_tminus4_oofv.std_errors[1],3)
    sm5 = round(mod_tminus5_oofv.std_errors[1],3)
    sm6 = round(mod_tminus1_oofv.std_errors[1],3)

    ####################################### results ###########################################
    table3_results["OOF projects"] = [tplus1_oofv, f"({sp1})", t_oofv, f"({s})", tminus1_oofv, f"({sm1})", tminus2_oofv, f"({sm2})", 
                                      tminus3_oofv, f"({sm3})", tminus4_oofv, f"({sm4})", tminus5_oofv, f"({sm5})", tminus6_oofv, f"({sm6})"]

    ################################# ODA projects ###################################
    ## t+1 all
    dependendFS = data_lagged.f1OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "Reserves*probOFn_oda","factor1*probOFn_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oda(t+1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oda(t+1)", "l1population_ln"]])
    mod_tplus1_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tplus1_oda = round(mod_tplus1_oda.params[1],3)
    ## t all
    dependendFS = data_lagged.OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l1Reserves*probOFn_oda","l1factor1*probOFn_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oda(t)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oda(t)", "l1population_ln"]])
    mod_t_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    t_oda = round(mod_t_oda.params[1],3)
    ## t -1 all
    dependendFS = data_lagged.l1OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l2Reserves*probOFn_oda","l2factor1*probOFn_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oda(t-1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oda(t-1)", "l1population_ln"]])
    mod_tminus1_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus1_oda = round(mod_tminus1_oda.params[1],3)
    ## t -2 all
    dependendFS = data_lagged.l2OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l3Reserves*probOFn_oda","l3factor1*probOFn_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oda(t-2)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oda(t-2)", "l1population_ln"]])
    mod_tminus2_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus2_oda = round(mod_tminus2_oda.params[1],3)
    ## t -3 all
    dependendFS = data_lagged.l3OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l4Reserves*probOFn_oda","l4factor1*probOFn_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oda(t-3)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oda(t-3)", "l1population_ln"]])
    mod_tminus3_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus3_oda = round(mod_tminus3_oda.params[1],3)
    ## t -4 all
    dependendFS = data_lagged.l4OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l5Reserves*probOFn_oda","l5factor1*probOFn_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oda(t-4)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oda(t-4)", "l1population_ln"]])
    mod_tminus4_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus4_oda = round(mod_tminus4_oda.params[1],3)
    ## t -5 all
    dependendFS = data_lagged.l5OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l6Reserves*probOFn_oda","l6factor1*probOFn_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oda(t-5)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oda(t-5)", "l1population_ln"]])
    mod_tminus5_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus5_oda = round(mod_tminus5_oda.params[1],3)
    ## t -6 all
    dependendFS = data_lagged.l6OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l7Reserves*probOFn_oda","l7factor1*probOFn_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFn_oda(t-6)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFn_oda(t-6)", "l1population_ln"]])
    mod_tminus6_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus6_oda = round(mod_tminus6_oda.params[1],3)
    
    sp1 = round(mod_tplus1_oda.std_errors[1],3)
    s = round(mod_t_oda.std_errors[1],3)
    sm2 = round(mod_tminus2_oda.std_errors[1],3)
    sm3 = round(mod_tminus3_oda.std_errors[1],3)
    sm4 = round(mod_tminus4_oda.std_errors[1],3)
    sm5 = round(mod_tminus5_oda.std_errors[1],3)
    sm6 = round(mod_tminus1_oda.std_errors[1],3)


    ####################################### results ###########################################
    table3_results["ODA projects"] = [tplus1_oda, f"({sp1})", t_oda, f"({s})", tminus1_oda, f"({sm1})", tminus2_oda, f"({sm2})", tminus3_oda, f"({sm3})", 
                                      tminus4_oda, f"({sm4})", tminus5_oda, f"({sm5})", tminus6_oda, f"({sm6})"]

    ################################ Project amounts ###################################
    ################################# All projects ###################################
    ## t+1 all
    dependendFS = data_lagged.f1OFa_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "Reserves*probOFa_all","factor1*probOFa_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa(t+1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa(t+1)", "l1population_ln"]])
    mod_tplus1_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tplus1_all = round(mod_tplus1_all.params[1],3)
    ## t all
    dependendFS = data_lagged.OFa_all_con_ln
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l1Reserves*probOFa_all","l1factor1*probOFa_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa(t)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa(t)", "l1population_ln"]])
    mod_t_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    t_all = round(mod_t_all.params[1],3)
    ## t -1 all
    dependendFS = data_lagged.l1OFa_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l2Reserves*probOFa_all","l2factor1*probOFa_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa(t-1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa(t-1)", "l1population_ln"]])
    mod_tminus1_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus1_all = round(mod_tminus1_all.params[1],3)
    ## t -2 all
    dependendFS = data_lagged.l2OFa_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l3Reserves*probOFa_all","l3factor1*probOFa_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa(t-2)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa(t-2)", "l1population_ln"]])
    mod_tminus2_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus2_all = round(mod_tminus2_all.params[1],3)
    ## t -3 all
    dependendFS = data_lagged.l3OFa_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l4Reserves*probOFa_all","l4factor1*probOFa_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa(t-3)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa(t-3)", "l1population_ln"]])
    mod_tminus3_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus3_all = round(mod_tminus3_all.params[1],3)
    ## t -4 all
    dependendFS = data_lagged.l4OFa_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l5Reserves*probOFa_all","l5factor1*probOFa_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa(t-4)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa(t-4)", "l1population_ln"]])
    mod_tminus4_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus4_all = round(mod_tminus4_all.params[1],3)
    ## t -5 all
    dependendFS = data_lagged.l5OFa_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l6Reserves*probOFa_all","l6factor1*probOFa_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa(t-5)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa(t-5)", "l1population_ln"]])
    mod_tminus5_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus5_all = round(mod_tminus5_all.params[1],3)
    ## t -6 all
    dependendFS = data_lagged.l6OFa_all
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l7Reserves*probOFa_all","l7factor1*probOFa_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa(t-6)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa(t-6)", "l1population_ln"]])
    mod_tminus6_all = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus6_all = round(mod_tminus6_all.params[1],3)
    
    sp1 = round(mod_tplus1_all.std_errors[1],3)
    s = round(mod_t_all.std_errors[1],3)
    sm2 = round(mod_tminus2_all.std_errors[1],3)
    sm3 = round(mod_tminus3_all.std_errors[1],3)
    sm4 = round(mod_tminus4_all.std_errors[1],3)
    sm5 = round(mod_tminus5_all.std_errors[1],3)
    sm6 = round(mod_tminus1_all.std_errors[1],3)

    ####################################### results ###########################################
    table3_results["(log) All amounts"] = [tplus1_all,f"({sp1})", t_all, f"({s})", tminus1_all, f"({sm1})", tminus2_all, f"({sm2})", tminus3_all, f"({sm3})",
                                           tminus4_all, f"({sm4})", tminus5_all, f"({sm5})", tminus6_all, f"({sm6})"]

    ################################# OOF projects ###################################
    ## t+1 all
    dependendFS = data_lagged.f1OFa_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "Reserves*probOFa_oofv","factor1*probOFa_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oofv(t+1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oofv(t+1)", "l1population_ln"]])
    mod_tplus1_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tplus1_oofv = round(mod_tplus1_oofv.params[1],3)
    ## t all
    dependendFS = data_lagged.OFa_oofv_con_ln
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l1Reserves*probOFa_oofv","l1factor1*probOFa_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oofv(t)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oofv(t)", "l1population_ln"]])
    mod_t_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    t_oofv = round(mod_t_oofv.params[1],3)
    ## t -1 all
    dependendFS = data_lagged.l1OFa_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l2Reserves*probOFa_oofv","l2factor1*probOFa_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oofv(t-1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oofv(t-1)", "l1population_ln"]])
    mod_tminus1_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus1_oofv = round(mod_tminus1_oofv.params[1],3)
    ## t -2 all
    dependendFS = data_lagged.l2OFa_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l3Reserves*probOFa_oofv","l3factor1*probOFa_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oofv(t-2)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oofv(t-2)", "l1population_ln"]])
    mod_tminus2_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus2_oofv = round(mod_tminus2_oofv.params[1],3)
    ## t -3 all
    dependendFS = data_lagged.l3OFa_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l4Reserves*probOFa_oofv","l4factor1*probOFa_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oofv(t-3)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oofv(t-3)", "l1population_ln"]])
    mod_tminus3_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus3_oofv = round(mod_tminus3_oofv.params[1],3)
    ## t -4 all
    dependendFS = data_lagged.l4OFa_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l5Reserves*probOFa_oofv","l5factor1*probOFa_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oofv(t-4)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oofv(t-4)", "l1population_ln"]])
    mod_tminus4_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus4_oofv = round(mod_tminus4_oofv.params[1],3)
    ## t -5 all
    dependendFS = data_lagged.l5OFa_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l6Reserves*probOFa_oofv","l6factor1*probOFa_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oofv(t-5)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oofv(t-5)", "l1population_ln"]])
    mod_tminus5_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus5_oofv = round(mod_tminus5_oofv.params[1],3)
    ## t -6 all
    dependendFS = data_lagged.l6OFa_oofv
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l7Reserves*probOFa_oofv","l7factor1*probOFa_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oofv(t-6)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oofv(t-6)", "l1population_ln"]])
    mod_tminus6_oofv = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus6_oofv = round(mod_tminus6_oofv.params[1],3)
    
    sp1 = round(mod_tplus1_oofv.std_errors[1],3)
    s = round(mod_t_oofv.std_errors[1],3)
    sm2 = round(mod_tminus2_oofv.std_errors[1],3)
    sm3 = round(mod_tminus3_oofv.std_errors[1],3)
    sm4 = round(mod_tminus4_oofv.std_errors[1],3)
    sm5 = round(mod_tminus5_oofv.std_errors[1],3)
    sm6 = round(mod_tminus1_oofv.std_errors[1],3)


    ####################################### results ###########################################
    table3_results["(log) OOF amounts"] = [tplus1_oofv, f"({sp1})", t_oofv, f"({s})", tminus1_oofv, f"({sm1})", tminus2_oofv, f"({sm2})", 
                                      tminus3_oofv, f"({sm3})", tminus4_oofv, f"({sm4})", tminus5_oofv, f"({sm5})", tminus6_oofv, f"({sm6})"]

    ################################# ODA projects ###################################
    ## t+1 all
    dependendFS = data_lagged.f1OFa_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "Reserves*probOFa_oda","factor1*probOFa_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oda(t+1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oda(t+1)", "l1population_ln"]])
    mod_tplus1_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tplus1_oda = round(mod_tplus1_oda.params[1],3)
    ## t all
    dependendFS = data_lagged.OFa_oda_con_ln
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l1Reserves*probOFa_oda","l1factor1*probOFa_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oda(t)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oda(t)", "l1population_ln"]])
    mod_t_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    t_oda = round(mod_t_oda.params[1],3)
    ## t -1 all
    dependendFS = data_lagged.l1OFa_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l2Reserves*probOFa_oda","l2factor1*probOFa_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oda(t-1)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oda(t-1)", "l1population_ln"]])
    mod_tminus1_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus1_oda = round(mod_tminus1_oda.params[1],3)
    ## t -2 all
    dependendFS = data_lagged.l2OFa_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l3Reserves*probOFa_oda","l3factor1*probOFa_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oda(t-2)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oda(t-2)", "l1population_ln"]])
    mod_tminus2_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus2_oda = round(mod_tminus2_oda.params[1],3)
    ## t -3 all
    dependendFS = data_lagged.l3OFa_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l4Reserves*probOFa_oda","l4factor1*probOFa_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oda(t-3)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oda(t-3)", "l1population_ln"]])
    mod_tminus3_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus3_oda = round(mod_tminus3_oda.params[1],3)
    ## t -4 all
    dependendFS = data_lagged.l4OFa_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l5Reserves*probOFa_oda","l5factor1*probOFa_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oda(t-4)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oda(t-4)", "l1population_ln"]])
    mod_tminus4_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus4_oda = round(mod_tminus4_oda.params[1],3)
    ## t -5 all
    dependendFS = data_lagged.l5OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l6Reserves*probOFa_oda","l6factor1*probOFa_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oda(t-5)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oda(t-5)", "l1population_ln"]])
    mod_tminus5_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus5_oda = round(mod_tminus5_oda.params[1],3)
    ## t -6 all
    dependendFS = data_lagged.l6OFn_oda
    exog2 = sm.tools.add_constant(data_lagged[["l1population_ln", "l7Reserves*probOFa_oda","l7factor1*probOFa_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data_lagged.code)

    fitted_c = mod_FS1.fitted_values
    data_lagged["Chinese_OFa_oda(t-6)"] = fitted_c

    dependentSS = data_lagged.growth_pc
    exog = sm.tools.add_constant(data_lagged[["Chinese_OFa_oda(t-6)", "l1population_ln"]])
    mod_tminus6_oda = lm.panel.PanelOLS(dependentSS, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data_lagged.code)
    tminus6_oda = round(mod_tminus6_oda.params[1],3)
    
    sp1 = round(mod_tplus1_oda.std_errors[1],3)
    s = round(mod_t_oda.std_errors[1],3)
    sm2 = round(mod_tminus2_oda.std_errors[1],3)
    sm3 = round(mod_tminus3_oda.std_errors[1],3)
    sm4 = round(mod_tminus4_oda.std_errors[1],3)
    sm5 = round(mod_tminus5_oda.std_errors[1],3)
    sm6 = round(mod_tminus1_oda.std_errors[1],3)

    ####################################### results ###########################################
    table3_results["(log) ODA amounts"] = [tplus1_oda, f"({sp1})", t_oda, f"({s})", tminus1_oda, f"({sm1})", tminus2_oda, f"({sm2})", tminus3_oda, f"({sm3})", 
                                           tminus4_oda, f"({sm4})", tminus5_oda, f"({sm5})", tminus6_oda, f"({sm6})"]
    table3_results["Observations"] = [mod_tplus1_oda.nobs, "-", mod_t_oda.nobs,"-", mod_tminus1_oda.nobs,"-", mod_tminus2_oda.nobs,"-",
                                      mod_tminus3_oda.nobs,"-", mod_tminus4_oda.nobs, "-",mod_tminus5_oda.nobs, "-",mod_tminus6_oda.nobs,"-"]
    return(table3_results)


####


def replicate_table5(data, alpha):
    
    table4_results = pd.DataFrame(index=["Panel A: Gross Fixed Capital Formation    ","SE", "p-value", "CI","",
                                         "Panel B: Gross Fixed Privat Formation     ","SE", "p-value", "CI","",
                                         "Panel C: Foreign Direct Investment Inflows","SE", "p-value", "CI","",
                                         "Panel D: Imports","SE","p-value", "CI","",
                                         "Panel E: Exports","SE","p-value", "CI","",
                                         "Panel F: Consumption, overall","SE","p-value", "CI","",
                                         "Panel G: Consumption, households","SE","p-value", "CI","",
                                         "Panel H: Consumption, government","SE","p-value", "CI","",
                                         "Panel I: Savings", "SE","p-value", "CI",""])
   # warnings.filterwarnings("ignore")

##### all regressions for all projects

    dependendFS = data.l2OFn_all
    exog2 = sm.tools.add_constant(data[["l1population_ln", "l3Reserves*probOFn_all","l3factor1*probOFn_all"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data.code)

    fitted_c = mod_FS1.fitted_values
    data["Chinese_OFn(t-2)"] = fitted_c
    

    dependentSS1 = data.dgfcf_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS1, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS1 = round(mod_all.params[1],3) 
    SE1 = round(mod_all.std_errors[1],3)
    PV1 = round(mod_all.pvalues[1],3)
    CI1 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS2 = data.dgfcf_priv_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS2, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS2 = round(mod_all.params[1],3) 
    SE2 = round(mod_all.pvalues[1],3)
    PV2 = round(mod_all.pvalues[1],3)
    CI2 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS3 = data.dfdi_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS3, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS3 = round(mod_all.params[1],3) 
    SE3 = round(mod_all.std_errors[1],3)
    PV3 = round(mod_all.pvalues[1],3)
    CI3 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS4 = data.dimp_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS4, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS4 = round(mod_all.params[1],3) 
    SE4 = round(mod_all.std_errors[1],3)
    PV4 = round(mod_all.pvalues[1],3)
    CI4 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS5 = data.dexp_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS5, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS5 = round(mod_all.params[1],3) 
    SE5 = round(mod_all.std_errors[1],3)
    PV5 = round(mod_all.pvalues[1],3)
    CI5 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS6 = data.dcons_all_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS6, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS6 = round(mod_all.params[1],3) 
    SE6 = round(mod_all.std_errors[1],3)
    PV6 = round(mod_all.pvalues[1],3)
    CI6 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS7 = data.dcons_hh_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS7, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS7 = round(mod_all.params[1],3) 
    SE7 = round(mod_all.std_errors[1],3)
    PV7 = round(mod_all.pvalues[1],3)
    CI7 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS8 = data.dcons_gov_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS8, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS8 = round(mod_all.params[1],3) 
    SE8 = round(mod_all.std_errors[1],3)
    PV8 = round(mod_all.pvalues[1],3)
    CI8 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS9 = data.dsav_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS9, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS9 = round(mod_all.params[1],3) 
    SE9 = round(mod_all.std_errors[1],3)
    PV9 = round(mod_all.pvalues[1],3)
    CI9 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    
    table4_results["All projects (OFn(t-2))"] = [SS1, f"({SE1})", PV1, CI1, "", SS2, f"({SE2})", PV2, CI2, "", SS3, f"({SE3})",
                                                 PV3, CI3, "", SS4, f"({SE4})", PV4, CI4,"", SS5,f"({SE5})",PV5,CI5, "",
                                                 SS6,f"({SE6})", PV6,CI6, "", SS7, f"({SE7})", PV7, CI7, "",
                                                 SS8, f"({SE8})", PV8, CI8,"", SS9, f"({SE9})", PV9,CI9, ""]
    
###################################################################################
###################################################################################

    dependendFS = data.l2OFn_oofv
    exog2 = sm.tools.add_constant(data[["l1population_ln", "l3Reserves*probOFn_oofv","l3factor1*probOFn_oofv"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data.code)

    fitted_c = mod_FS1.fitted_values
    data["Chinese_OFn(t-2)"] = fitted_c
    

    dependentSS1 = data.dgfcf_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS1, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS1 = round(mod_all.params[1],3) 
    SE1 = round(mod_all.std_errors[1],3)
    PV1 = round(mod_all.pvalues[1],3)
    CI1 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS2 = data.dgfcf_priv_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS2, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS2 = round(mod_all.params[1],3) 
    SE2 = round(mod_all.pvalues[1],3)
    PV2 = round(mod_all.pvalues[1],3)
    CI2 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS3 = data.dfdi_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS3, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS3 = round(mod_all.params[1],3) 
    SE3 = round(mod_all.std_errors[1],3)
    PV3 = round(mod_all.pvalues[1],3)
    CI3 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS4 = data.dimp_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS4, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS4 = round(mod_all.params[1],3) 
    SE4 = round(mod_all.std_errors[1],3)
    PV4 = round(mod_all.pvalues[1],3)
    CI4 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS5 = data.dexp_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS5, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS5 = round(mod_all.params[1],3) 
    SE5 = round(mod_all.std_errors[1],3)
    PV5 = round(mod_all.pvalues[1],3)
    CI5 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS6 = data.dcons_all_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS6, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS6 = round(mod_all.params[1],3) 
    SE6 = round(mod_all.std_errors[1],3)
    PV6 = round(mod_all.pvalues[1],3)
    CI6 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS7 = data.dcons_hh_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS7, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS7 = round(mod_all.params[1],3) 
    SE7 = round(mod_all.std_errors[1],3)
    PV7 = round(mod_all.pvalues[1],3)
    CI7 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS8 = data.dcons_gov_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS8, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS8 = round(mod_all.params[1],3) 
    SE8 = round(mod_all.std_errors[1],3)
    PV8 = round(mod_all.pvalues[1],3)
    CI8 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    dependentSS9 = data.dsav_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS9, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS9 = round(mod_all.params[1],3) 
    SE9 = round(mod_all.std_errors[1],3)
    PV9 = round(mod_all.pvalues[1],3)
    CI9 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    
    
    table4_results["OOFV projects (OFn(t-2))"] = [SS1, f"({SE1})", PV1, CI1, "", SS2, f"({SE2})", PV2, CI2, "", SS3, f"({SE3})",
                                                 PV3, CI3, "", SS4, f"({SE4})", PV4, CI4,"", SS5,f"({SE5})",PV5,CI5, "",
                                                 SS6,f"({SE6})", PV6,CI6, "", SS7, f"({SE7})", PV7, CI7, "",
                                                 SS8, f"({SE8})", PV8, CI8,"", SS9, f"({SE9})", PV9,CI9, ""]
    
###################################################################################
###################################################################################

    dependendFS = data.l2OFn_oda
    exog2 = sm.tools.add_constant(data[["l1population_ln", "l3Reserves*probOFn_oda","l3factor1*probOFn_oda"]])
    mod_FS1 = lm.panel.PanelOLS(dependendFS, exog2, time_effects = True, entity_effects=True, drop_absorbed=True).fit(cov_type='clustered', clusters = data.code)

    fitted_c = mod_FS1.fitted_values
    data["Chinese_OFn(t-2)"] = fitted_c
    

    dependentSS1 = data.dgfcf_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS1, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS1 = round(mod_all.params[1],3) 
    SE1 = round(mod_all.std_errors[1],3)
    PV1 = round(mod_all.pvalues[1],3)
    CI1 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    n1 = mod_all.nobs
    
    dependentSS2 = data.dgfcf_priv_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS2, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS2 = round(mod_all.params[1],3) 
    SE2 = round(mod_all.pvalues[1],3)
    PV2 = round(mod_all.pvalues[1],3)
    CI2 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    n2 = mod_all.nobs
    
    dependentSS3 = data.dfdi_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS3, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS3 = round(mod_all.params[1],3) 
    SE3 = round(mod_all.std_errors[1],3)
    PV3 = round(mod_all.pvalues[1],3)
    CI3 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    n3 = mod_all.nobs
    
    dependentSS4 = data.dimp_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS4, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS4 = round(mod_all.params[1],3) 
    SE4 = round(mod_all.std_errors[1],3)
    PV4 = round(mod_all.pvalues[1],3)
    CI4 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    n4 = mod_all.nobs
    
    dependentSS5 = data.dexp_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS5, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS5 = round(mod_all.params[1],3) 
    SE5 = round(mod_all.std_errors[1],3)
    PV5 = round(mod_all.pvalues[1],3)
    CI5 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    n5 = mod_all.nobs
    
    dependentSS6 = data.dcons_all_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS6, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS6 = round(mod_all.params[1],3) 
    SE6 = round(mod_all.std_errors[1],3)
    PV6 = round(mod_all.pvalues[1],3)
    CI6 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    n6 = mod_all.nobs
    
    dependentSS7 = data.dcons_hh_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS7, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS7 = round(mod_all.params[1],3) 
    SE7 = round(mod_all.std_errors[1],3)
    PV7 = round(mod_all.pvalues[1],3)
    CI7 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    n7 = mod_all.nobs
    
    dependentSS8 = data.dcons_gov_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS8, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS8 = round(mod_all.params[1],3) 
    SE8 = round(mod_all.std_errors[1],3)
    PV8 = round(mod_all.pvalues[1],3)
    CI8 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    n8 = mod_all.nobs
    
    dependentSS9 = data.dsav_con_ln
    exog = sm.tools.add_constant(data[["Chinese_OFn(t-2)", "l1population_ln"]])
    mod_all = lm.panel.PanelOLS(dependentSS9, exog, time_effects=True, entity_effects= True).fit(cov_type='clustered', clusters = data.code)
    SS9 = round(mod_all.params[1],3) 
    SE9 = round(mod_all.std_errors[1],3)
    PV9 = round(mod_all.pvalues[1],3)
    CI9 = round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)","lower"],4), round(mod_all.conf_int(1-alpha).loc["Chinese_OFn(t-2)", "upper"],4)
    n9 = mod_all.nobs
    
    table4_results["ODA projects (OFn(t-2))"] = [SS1, f"({SE1})", PV1, CI1, "", SS2, f"({SE2})", PV2, CI2, "", SS3, f"({SE3})",
                                                 PV3, CI3, "", SS4, f"({SE4})", PV4, CI4,"", SS5,f"({SE5})",PV5,CI5, "",
                                                 SS6,f"({SE6})", PV6,CI6, "", SS7, f"({SE7})", PV7, CI7, "",
                                                 SS8, f"({SE8})", PV8, CI8,"", SS9, f"({SE9})", PV9,CI9, ""]
    #number of obs
    table4_results["number Obs."] = [n1,"","","","",n2,"","","","",n3,"","","","",n4,"","","","",n5,"","","","",
                              n6,"","","","",n7,"","","","",n8,"","","","",n9,"","","",""]
    
    # how many countries were considered?
    table4_results["countries"] = [(150-data.dgfcf_con_ln.groupby(data.code).sum().value_counts()[0]),"","","","",
                                 (150-data.dgfcf_priv_con_ln.groupby(data.code).sum().value_counts()[0]),"","","","",
                                  (150-data.dfdi_con_ln.groupby(data.code).sum().value_counts()[0]),"","","","",
                                  (150-data.dimp_con_ln.groupby(data.code).sum().value_counts()[0]),"","","","",
                                  (150-data.dexp_con_ln.groupby(data.code).sum().value_counts()[0]),"","","","",
                                  (150-data.dcons_all_con_ln.groupby(data.code).sum().value_counts()[0]),"","","","",
                                  (150-data.dcons_hh_con_ln.groupby(data.code).sum().value_counts()[0]),"","","","",
                                  (150-data.dcons_gov_con_ln.groupby(data.code).sum().value_counts()[0]),"","","","",
                                  (150-data.dsav_con_ln.groupby(data.code).sum().value_counts()[0]),"","","",""]
    
    
    print("Confidence Intervall reported for alpha:", alpha)                              
    return(table4_results)
   

