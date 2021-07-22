"""This module contains auxiliary functions for the preparation and modification of the data."""

import numpy as np
import pandas as pd
import pandas.io.formats.style
import seaborn as sns
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm_api
import matplotlib.pyplot as plt
import geopandas 
import itertools as tools
from IPython.display import HTML


def get_allocation_data():
    # Data preperation for the following part
    data3 = pd.read_stata("data/AEJ2020 allocation.dta")
    ######

    # prepare data
    # we have to include 1 year lag and 3 year lagg
    # since we have panel data, skip(does not work so we have to do it manually)
    years = ["2000-01-01","2001-01-01","2002-01-01","2003-01-01","2004-01-01","2005-01-01","2006-01-01","2007-01-01",
              "2008-01-01","2009-01-01","2010-01-01","2011-01-01","2012-01-01","2013-01-01", "2014-01-01"]

    # factor1 : 0,452441 = 1997
    factors = data3.factor1.unique()
    reserves = data3.det_reservesCHN_con.unique()

    data3["factor1_1"] = data3.factor1.shift(1)
    #data3["factor1_3"] = np.nan, data4["OFa_oda_con_ln"].shift(2)
    data3["det_reservesCHN_con_1"] = data3.det_reservesCHN_con.shift(1)
    

    # Now Drop missing values and return to 2000-2014 period
    data3 = data3[data3.year >= "2000-01-01"].copy()
    data3.replace([np.inf, -np.inf], np.nan, inplace=True);
    data3 = data3.dropna(0, subset = ["A_LINLINECHN","A_Ltaiwanr","A_Ltrade_con_ln","A_D99petroleum",
                 "A_LDebtGDP","A_Lpolity2","A_Lgdppc_con_ln", "A_Lpopulation_ln","A_Lenglish"])

    
    return(data3)

###

def get_effectiveness_data(multiindex):
    data4 = pd.read_stata("data/AEJ2020 effectiveness.dta")

    data4 = data4[data4.year >= "1995-01-01"].copy()
    data4 = data4.set_index(["countryname", "year"])

    # First I will check and create rows for all countries that miss some years of observation. 
    # This is done in order to use tje shift() function properly
    # Without this, shift does not work properly.

    idx = list(tools.product(data4.index.levels[0], data4.index.levels[1]))
    data4 = data4.reindex(idx).reset_index()

    # reset the index again and apply the lagged transformations 

    data4["l3factor1"] = data4.factor1.shift(3)
    data4["l3Reserves"] = data4.reservesCHN_con.shift(3)
    data4["l1population_ln"] = data4['population_ln'].shift(1)
    data4["l2OFn_all"] = data4['OFn_all'].shift(2)
    data4["l2OFn_oofv"] = data4["OFn_oofv"].shift(2)
    data4["l2OFn_oda"] = data4["OFn_oda"].shift(2)
    
    
    # rename variable
    
    data4["l2OFa_all_ln"] = data4.OFa_all_con_ln.shift(2)
    data4["l2OFa_oofv_ln"] = data4["OFa_oofv_con_ln"].shift(2)
    data4["l2OFa_oda_ln"] = data4["OFa_oda_con_ln"].shift(2)
    
    data4["l3Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"].shift(3)
    data4["l3factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"].shift(3)
    
    data4["l3Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"].shift(3)
    data4["l3factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"].shift(3)
    
    data4["l3Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"].shift(3)
    data4["l3factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"].shift(3)
    
    data4["l3Reserves*probOFa_all_ln"] = data4["IV_reserves_OFa_all_1_ln"].shift(3)
    data4["l3factor1*probOFa_all_ln"] = data4["IV_factor1_OFa_all_1_ln"].shift(3)
    
    data4["l3Reserves*probOFa_oda_ln"] = data4["IV_reserves_OFa_oda_1_ln"].shift(3)
    data4["l3factor1*probOFa_oda_ln"] = data4["IV_factor1_OFa_oda_1_ln"].shift(3)
    
    data4["l3Reserves*probOFa_oofv_ln"] = data4["IV_reserves_OFa_oofv_1_ln"].shift(3)
    data4["l3factor1*probOFa_oofv_ln"] = data4["IV_factor1_OFa_oofv_1_ln"].shift(3)
    
    # supress special type of warnings
    pd.options.mode.chained_assignment = None  
    
    # variables for robustness Checks
    data4["l2Exports_ln"] = np.log(data4.exports_china+1).shift(2);
    helperFDI = data4.ifdi_from_china;
    
    # Getting year-to-year differences for Table 4 by calculating tis years number minus last years numbers
    # Gross Fixed Capital Formation
    data4["l1gfcf_con_ln"] = data4['gfcf_con_ln'].shift(1)   
    data4["dgfcf_con_ln"] = data4['gfcf_con_ln'] - data4["l1gfcf_con_ln"] 
    # Gross Fixed Private Capital Formation
    data4["l1gfcf_priv_con_ln"] = data4['gfcf_priv_con_ln'].shift(1)   
    data4["dgfcf_priv_con_ln"] = data4['gfcf_priv_con_ln'] - data4["l1gfcf_priv_con_ln"] 
    # Imports
    data4["l1imp_con_ln"] = data4['imp_con_ln'].shift(1)   
    data4["dimp_con_ln"] = data4['imp_con_ln'] - data4["l1imp_con_ln"] 
    # Exports
    data4["l1exp_con_ln"] = data4['exp_con_ln'].shift(1)   
    data4["dexp_con_ln"] = data4['exp_con_ln'] - data4["l1exp_con_ln"]  
    # Overall Consumption
    data4["l1cons_all_con_ln"] = data4['cons_all_con_ln'].shift(1)   
    data4["dcons_all_con_ln"] = data4['cons_all_con_ln'] - data4["l1cons_all_con_ln"] 
    # Houshold Consumption
    data4["l1cons_hh_con_ln"] = data4['cons_hh_con_ln'].shift(1)   
    data4["dcons_hh_con_ln"] = data4['cons_hh_con_ln'] - data4["l1cons_hh_con_ln"] 
    # Government Consumption
    data4["l1cons_gov_con_ln"] = data4['cons_gov_con_ln'].shift(1)   
    data4["dcons_gov_con_ln"] = data4['cons_gov_con_ln'] - data4["l1cons_gov_con_ln"] 
    # Savings
    data4["l1sav_con_ln"] = data4['sav_con_ln'].shift(1)   
    data4["dsav_con_ln"] = data4['sav_con_ln'] - data4["l1sav_con_ln"] 
    # FDI Inflow
    data4["l1fdi_con_ln"] = data4['fdi_con_ln'].shift(1)   
    data4["dfdi_con_ln"] = data4['fdi_con_ln'] - data4["l1fdi_con_ln"] 
    
    
    #clean data
    helperFDI.loc[helperFDI <= 0] = 0;
    helperFDI = np.log(helperFDI+1);
    data4["l2FDI_China_ln"] = helperFDI.shift(2);
  
    
    # Set observations for countries to NaN according to Appendix C1 of the paper
    data4.loc[data4.countryname == "Antigua and Barbuda"] = np.nan #only one observation
    data4.loc[data4.countryname == "China"] = np.nan               #not interested in impact on China
    data4.loc[data4.countryname == "Barbados"] = np.nan            #only one observation
    
    # apply final cutoff and cleaning up
    data4 = data4[data4.year >= "2000-01-01"].copy()
    data4.replace([np.inf, -np.inf], np.nan, inplace=True)
    data4 = data4.dropna(0, subset = ["l1population_ln", "l2OFn_all", "growth_pc"])#, "l2FDI_China_ln", "l2Exports_ln"])
    
    
    

    if multiindex == True:
        data4 = data4.set_index(["countryname", "year"])
    
    return(data4)

####

def get_effectiveness_data2(multiindex):
    data4 = pd.read_stata("data/AEJ2020 effectiveness.dta")

    data4 = data4[data4.year >= "1995-01-01"].copy()
    data4 = data4.set_index(["countryname", "year"])

    # First I will check and create rows for all countries that miss some years of observation. 
    # This is done in order to use tje shift() function properly
    # Without this, shift does not work properly.

    idx = list(tools.product(data4.index.levels[0], data4.index.levels[1]))
    data4 = data4.reindex(idx).reset_index()

    # reset the index again and apply the lagged transformations 

    data4["l3factor1"] = data4.factor1.shift(3)
    data4["l3Reserves"] = data4.reservesCHN_con.shift(3)
    data4["l1population_ln"] = data4['population_ln'].shift(1)
    data4["l2OFn_all"] = data4['OFn_all'].shift(2)
    data4["l2OFn_oofv"] = data4["OFn_oofv"].shift(2)
    data4["l2OFn_oda"] = data4["OFn_oda"].shift(2)
    
    
    # rename variable
    
    data4["l2OFa_all_ln"] = data4.OFa_all_con_ln.shift(2)
    data4["l2OFa_oofv_ln"] = data4["OFa_oofv_con_ln"].shift(2)
    data4["l2OFa_oda_ln"] = data4["OFa_oda_con_ln"].shift(2)
    
    data4["l3Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"].shift(3)
    data4["l3factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"].shift(3)
    
    data4["l3Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"].shift(3)
    data4["l3factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"].shift(3)
    
    data4["l3Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"].shift(3)
    data4["l3factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"].shift(3)
    
    data4["l3Reserves*probOFa_all_ln"] = data4["IV_reserves_OFa_all_1_ln"].shift(3)
    data4["l3factor1*probOFa_all_ln"] = data4["IV_factor1_OFa_all_1_ln"].shift(3)
    
    data4["l3Reserves*probOFa_oda_ln"] = data4["IV_reserves_OFa_oda_1_ln"].shift(3)
    data4["l3factor1*probOFa_oda_ln"] = data4["IV_factor1_OFa_oda_1_ln"].shift(3)
    
    data4["l3Reserves*probOFa_oofv_ln"] = data4["IV_reserves_OFa_oofv_1_ln"].shift(3)
    data4["l3factor1*probOFa_oofv_ln"] = data4["IV_factor1_OFa_oofv_1_ln"].shift(3)
    
    # supress special type of warnings
    pd.options.mode.chained_assignment = None  
    
    # variables for robustness Checks
    data4["l2Exports_ln"] = np.log(data4.exports_china+1).shift(2);
    helperFDI = data4.ifdi_from_china;
    
    # Getting year-to-year differences for Table 4 by calculating tis years number minus last years numbers
    # Gross Fixed Capital Formation
    data4["l1gfcf_con_ln"] = data4['gfcf_con_ln'].shift(1)   
    data4["dgfcf_con_ln"] = data4['gfcf_con_ln'] - data4["l1gfcf_con_ln"] 
    # Gross Fixed Private Capital Formation
    data4["l1gfcf_priv_con_ln"] = data4['gfcf_priv_con_ln'].shift(1)   
    data4["dgfcf_priv_con_ln"] = data4['gfcf_priv_con_ln'] - data4["l1gfcf_priv_con_ln"] 
    # Imports
    data4["l1imp_con_ln"] = data4['imp_con_ln'].shift(1)   
    data4["dimp_con_ln"] = data4['imp_con_ln'] - data4["l1imp_con_ln"] 
    # Exports
    data4["l1exp_con_ln"] = data4['exp_con_ln'].shift(1)   
    data4["dexp_con_ln"] = data4['exp_con_ln'] - data4["l1exp_con_ln"]  
    # Overall Consumption
    data4["l1cons_all_con_ln"] = data4['cons_all_con_ln'].shift(1)   
    data4["dcons_all_con_ln"] = data4['cons_all_con_ln'] - data4["l1cons_all_con_ln"] 
    # Houshold Consumption
    data4["l1cons_hh_con_ln"] = data4['cons_hh_con_ln'].shift(1)   
    data4["dcons_hh_con_ln"] = data4['cons_hh_con_ln'] - data4["l1cons_hh_con_ln"] 
    # Government Consumption
    data4["l1cons_gov_con_ln"] = data4['cons_gov_con_ln'].shift(1)   
    data4["dcons_gov_con_ln"] = data4['cons_gov_con_ln'] - data4["l1cons_gov_con_ln"] 
    # Savings
    data4["l1sav_con_ln"] = data4['sav_con_ln'].shift(1)   
    data4["dsav_con_ln"] = data4['sav_con_ln'] - data4["l1sav_con_ln"] 
    # FDI Inflow
    data4["l1fdi_con_ln"] = data4['fdi_con_ln'].shift(1)   
    data4["dfdi_con_ln"] = data4['fdi_con_ln'] - data4["l1fdi_con_ln"] 
    
    
    #clean data
    helperFDI.loc[helperFDI <= 0] = 0;
    helperFDI = np.log(helperFDI+1);
    data4["l2FDI_China_ln"] = helperFDI.shift(2);
  
    
    # Set observations for countries to NaN according to Appendix C1 of the paper
    data4.loc[data4.countryname == "Antigua and Barbuda"] = np.nan #only one observation
    data4.loc[data4.countryname == "China"] = np.nan               #not interested in impact on China
    data4.loc[data4.countryname == "Barbados"] = np.nan            #only one observation
    
    # apply final cutoff and cleaning up
    data4 = data4[data4.year >= "2000-01-01"].copy()
    data4.replace([np.inf, -np.inf], np.nan, inplace=True)
    data4 = data4.dropna(0, subset = ["l1population_ln"])#, "l2FDI_China_ln", "l2Exports_ln"])
    
    
    

    if multiindex == True:
        data4 = data4.set_index(["countryname", "year"])
    
    return(data4)




####

def get_parallel_trends_df(data4):
    years = ["2002-01-01","2003-01-01","2004-01-01","2005-01-01","2006-01-01","2007-01-01",
             "2008-01-01","2009-01-01","2010-01-01","2011-01-01","2012-01-01","2013-01-01", "2014-01-01"]

    results_df = pd.DataFrame(columns = ["year", "lower_probOFn_ln", "upper_probOFn_ln",
                                         "lower_probOFa_ln", "upper_probOFa_ln",
                                         "lower_probGrowthpc", "upper_probGrowthpc"])
    count = 0
    for year in years:
        df_help = data4[data4.year == year]
        #print(year)
        # sort probabilities of reeiving finance
        df_help = df_help.sort_values(by= "probaid_PRC_OFn_all", ascending = True)
        #define lower and upper
        lower_OFn = np.log(df_help.l2OFn_all[df_help.probaid_PRC_OFn_all < 0.66666667].mean())
        upper_OFn = np.log(df_help.l2OFn_all[df_help.probaid_PRC_OFn_all >= 0.66666667].mean())

        lower_Growth = df_help.growth_pc[df_help.probaid_PRC_OFn_all < 0.66666667].mean()
        upper_Growth = df_help.growth_pc[df_help.probaid_PRC_OFn_all >= 0.66666667].mean()

        lower_OFa = np.log(df_help.l2OFa_all_ln[df_help.probaid_PRC_OFn_all < 0.66666667].mean())
        upper_OFa = np.log(df_help.l2OFa_all_ln[df_help.probaid_PRC_OFn_all >= 0.66666667].mean())


        results_df.loc[count] = [year,lower_OFn,upper_OFn, lower_OFa, upper_OFa, lower_Growth,upper_Growth]
        count +=1

    return(results_df)


###

def get_effectiveness_data_various_lags(multiindex):
    data4 = pd.read_stata("data/AEJ2020 effectiveness.dta")

    data4 = data4[data4.year >= "1990-01-01"].copy()
    data4 = data4.set_index(["countryname", "year"])

    # First I will check and create rows for all countries that miss some years of observation. 
    # This is done in order to use tje shift() function properly
    # Without this, shift does not work properly.

    idx = list(tools.product(data4.index.levels[0], data4.index.levels[1]))
    data4 = data4.reindex(idx).reset_index()

    # reset the index again and apply the lagged transformations 

    #data4["l3factor1"] = data4.factor1.shift(3)
    #data4["l3Reserves"] = data4.reservesCHN_con.shift(3)
    #data4["l1population_ln"] = data4['population_ln'].shift(1)
    #data4["l2OFn_all"] = data4['OFn_all'].shift(2)
    #data4["l2OFn_oofv"] = data4["OFn_oofv"].shift(2)
    #data4["l2OFn_oda"] = data4["OFn_oda"].shift(2)
    data4["l1population_ln"] = data4['population_ln'].shift(1)
### variuos laggs project counts
# -1
    data4["f1OFn_all"] = data4['OFn_all'].shift(-1)
    data4["Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"]
    data4["factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"]
    data4["f1OFn_oofv"] = data4['OFn_oofv'].shift(-1) #oofv
    data4["Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"]
    data4["factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"]
    data4["f1OFn_oda"] = data4['OFn_oda'].shift(-1) #oda
    data4["Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"]
    data4["factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"]
# 0
    #data4["f1OFn_all"] = data4['OFn_all'].shift(-1)
    data4["l1Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"].shift(1)
    data4["l1factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"].shift(1)
    #data4["f1OFn_oofv"] = data4['OFn_oofv'].shift(-1) #oofv
    data4["l1Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"].shift(1)
    data4["l1factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"].shift(1)
    #data4["f1OFn_oda"] = data4['OFn_oda'].shift(-1) #oda
    data4["l1Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"].shift(1)
    data4["l1factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"].shift(1)
# +1
    data4["l1OFn_all"] = data4['OFn_all'].shift(1)
    data4["l2Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"].shift(2)
    data4["l2factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"].shift(2)
    data4["l1OFn_oofv"] = data4['OFn_oofv'].shift(1) #oofv
    data4["l2Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"].shift(2)
    data4["l2factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"].shift(2)
    data4["l1OFn_oda"] = data4['OFn_oda'].shift(1) #oda
    data4["l2Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"].shift(2)
    data4["l2factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"].shift(2)
# +2, base setup
    data4["l2OFn_all"] = data4['OFn_all'].shift(2)
    data4["l3Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"].shift(3)
    data4["l3factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"].shift(3)
    data4["l2OFn_oofv"] = data4['OFn_oofv'].shift(2) #oofv
    data4["l3Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"].shift(3)
    data4["l3factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"].shift(3)
    data4["l2OFn_oda"] = data4['OFn_oda'].shift(2) #oda
    data4["l3Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"].shift(3)
    data4["l3factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"].shift(3)
# +3
    data4["l3OFn_all"] = data4['OFn_all'].shift(3)
    data4["l4Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"].shift(4)
    data4["l4factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"].shift(4)
    data4["l3OFn_oofv"] = data4['OFn_oofv'].shift(3) #oofv
    data4["l4Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"].shift(4)
    data4["l4factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"].shift(4)
    data4["l3OFn_oda"] = data4['OFn_oda'].shift(3) #oda
    data4["l4Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"].shift(4)
    data4["l4factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"].shift(4)
# +4
    data4["l4OFn_all"] = data4['OFn_all'].shift(4)
    data4["l5Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"].shift(5)
    data4["l5factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"].shift(5)
    data4["l4OFn_oofv"] = data4['OFn_oofv'].shift(4) #oofv
    data4["l5Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"].shift(5)
    data4["l5factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"].shift(5)
    data4["l4OFn_oda"] = data4['OFn_oda'].shift(4) #oda
    data4["l5Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"].shift(5)
    data4["l5factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"].shift(5)
# +5
    data4["l5OFn_all"] = data4['OFn_all'].shift(5)
    data4["l6Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"].shift(6)
    data4["l6factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"].shift(6)
    data4["l5OFn_oofv"] = data4['OFn_oofv'].shift(5) #oofv
    data4["l6Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"].shift(6)
    data4["l6factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"].shift(6)
    data4["l5OFn_oda"] = data4['OFn_oda'].shift(5) #oda
    data4["l6Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"].shift(6)
    data4["l6factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"].shift(6)
# +6
    data4["l6OFn_all"] = data4['OFn_all'].shift(6)
    data4["l7Reserves*probOFn_all"] = data4["IV_reserves_OFn_all_1_ln"].shift(7)
    data4["l7factor1*probOFn_all"] = data4["IV_factor1_OFn_all_1_ln"].shift(7)
    data4["l6OFn_oofv"] = data4['OFn_oofv'].shift(6) #oofv
    data4["l7Reserves*probOFn_oofv"] = data4["IV_reserves_OFn_oofv_1_ln"].shift(7)
    data4["l7factor1*probOFn_oofv"] = data4["IV_factor1_OFn_oofv_1_ln"].shift(7)
    data4["l6OFn_oda"] = data4['OFn_oda'].shift(6) #oda
    data4["l7Reserves*probOFn_oda"] = data4["IV_reserves_OFn_oda_1_ln"].shift(7)
    data4["l7factor1*probOFn_oda"] = data4["IV_factor1_OFn_oda_1_ln"].shift(7)
    
#########################################

### variuos laggs amounts
# -1
    data4["f1OFa_all"] = data4['OFa_all_con_ln'].shift(-1)
    data4["Reserves*probOFa_all"] = data4["IV_reserves_OFa_all_1_ln"]
    data4["factor1*probOFa_all"] = data4["IV_factor1_OFa_all_1_ln"]
    data4["f1OFa_oofv"] = data4['OFa_oofv_con_ln'].shift(-1) #oofv
    data4["Reserves*probOFa_oofv"] = data4["IV_reserves_OFa_oofv_1_ln"]
    data4["factor1*probOFa_oofv"] = data4["IV_factor1_OFa_oofv_1_ln"]
    data4["f1OFa_oda"] = data4['OFa_oda_con_ln'].shift(-1) #oda
    data4["Reserves*probOFa_oda"] = data4["IV_reserves_OFa_oda_1_ln"]
    data4["factor1*probOFa_oda"] = data4["IV_factor1_OFa_oda_1_ln"]
# 0
    #data4["f1OFn_all"] = data4['OFn_all'].shift(-1)
    data4["l1Reserves*probOFa_all"] = data4["IV_reserves_OFa_all_1_ln"].shift(1)
    data4["l1factor1*probOFa_all"] = data4["IV_factor1_OFa_all_1_ln"].shift(1)
    #data4["f1OFn_oofv"] = data4['OFn_oofv'].shift(-1) #oofv
    data4["l1Reserves*probOFa_oofv"] = data4["IV_reserves_OFa_oofv_1_ln"].shift(1)
    data4["l1factor1*probOFa_oofv"] = data4["IV_factor1_OFa_oofv_1_ln"].shift(1)
    #data4["f1OFn_oda"] = data4['OFn_oda'].shift(-1) #oda
    data4["l1Reserves*probOFa_oda"] = data4["IV_reserves_OFa_oda_1_ln"].shift(1)
    data4["l1factor1*probOFa_oda"] = data4["IV_factor1_OFa_oda_1_ln"].shift(1)
# +1
    data4["l1OFa_all"] = data4['OFa_all_con_ln'].shift(1)
    data4["l2Reserves*probOFa_all"] = data4["IV_reserves_OFa_all_1_ln"].shift(2)
    data4["l2factor1*probOFa_all"] = data4["IV_factor1_OFa_all_1_ln"].shift(2)
    data4["l1OFa_oofv"] = data4['OFa_oofv_con_ln'].shift(1) #oofv
    data4["l2Reserves*probOFa_oofv"] = data4["IV_reserves_OFa_oofv_1_ln"].shift(2)
    data4["l2factor1*probOFa_oofv"] = data4["IV_factor1_OFa_oofv_1_ln"].shift(2)
    data4["l1OFa_oda"] = data4['OFa_oda_con_ln'].shift(1) #oda
    data4["l2Reserves*probOFa_oda"] = data4["IV_reserves_OFa_oda_1_ln"].shift(2)
    data4["l2factor1*probOFa_oda"] = data4["IV_factor1_OFa_oda_1_ln"].shift(2)
# +2, base setup
    data4["l2OFa_all"] = data4['OFa_all_con_ln'].shift(2)
    data4["l3Reserves*probOFa_all"] = data4["IV_reserves_OFa_all_1_ln"].shift(3)
    data4["l3factor1*probOFa_all"] = data4["IV_factor1_OFa_all_1_ln"].shift(3)
    data4["l2OFa_oofv"] = data4['OFa_oofv_con_ln'].shift(2) #oofv
    data4["l3Reserves*probOFa_oofv"] = data4["IV_reserves_OFa_oofv_1_ln"].shift(3)
    data4["l3factor1*probOFa_oofv"] = data4["IV_factor1_OFa_oofv_1_ln"].shift(3)
    data4["l2OFa_oda"] = data4['OFa_oda_con_ln'].shift(2) #oda
    data4["l3Reserves*probOFa_oda"] = data4["IV_reserves_OFa_oda_1_ln"].shift(3)
    data4["l3factor1*probOFa_oda"] = data4["IV_factor1_OFa_oda_1_ln"].shift(3)
# +3
    data4["l3OFa_all"] = data4['OFa_all_con_ln'].shift(3)
    data4["l4Reserves*probOFa_all"] = data4["IV_reserves_OFa_all_1_ln"].shift(4)
    data4["l4factor1*probOFa_all"] = data4["IV_factor1_OFa_all_1_ln"].shift(4)
    data4["l3OFa_oofv"] = data4['OFa_oofv_con_ln'].shift(3) #oofv
    data4["l4Reserves*probOFa_oofv"] = data4["IV_reserves_OFa_oofv_1_ln"].shift(4)
    data4["l4factor1*probOFa_oofv"] = data4["IV_factor1_OFa_oofv_1_ln"].shift(4)
    data4["l3OFa_oda"] = data4['OFa_oda_con_ln'].shift(3) #oda
    data4["l4Reserves*probOFa_oda"] = data4["IV_reserves_OFa_oda_1_ln"].shift(4)
    data4["l4factor1*probOFa_oda"] = data4["IV_factor1_OFa_oda_1_ln"].shift(4)
# +4
    data4["l4OFa_all"] = data4['OFa_all_con_ln'].shift(4)
    data4["l5Reserves*probOFa_all"] = data4["IV_reserves_OFa_all_1_ln"].shift(5)
    data4["l5factor1*probOFa_all"] = data4["IV_factor1_OFa_all_1_ln"].shift(5)
    data4["l4OFa_oofv"] = data4['OFa_oofv_con_ln'].shift(4) #oofv
    data4["l5Reserves*probOFa_oofv"] = data4["IV_reserves_OFa_oofv_1_ln"].shift(5)
    data4["l5factor1*probOFa_oofv"] = data4["IV_factor1_OFa_oofv_1_ln"].shift(5)
    data4["l4OFa_oda"] = data4['OFa_oda_con_ln'].shift(4) #oda
    data4["l5Reserves*probOFa_oda"] = data4["IV_reserves_OFa_oda_1_ln"].shift(5)
    data4["l5factor1*probOFa_oda"] = data4["IV_factor1_OFa_oda_1_ln"].shift(5)
# +5
    data4["l5OFa_all"] = data4['OFa_all_con_ln'].shift(5)
    data4["l6Reserves*probOFa_all"] = data4["IV_reserves_OFa_all_1_ln"].shift(6)
    data4["l6factor1*probOFa_all"] = data4["IV_factor1_OFa_all_1_ln"].shift(6)
    data4["l5OFa_oofv"] = data4['OFa_oofv_con_ln'].shift(5) #oofv
    data4["l6Reserves*probOFa_oofv"] = data4["IV_reserves_OFa_oofv_1_ln"].shift(6)
    data4["l6factor1*probOFa_oofv"] = data4["IV_factor1_OFa_oofv_1_ln"].shift(6)
    data4["l5OFa_oda"] = data4['OFa_oda_con_ln'].shift(5) #oda
    data4["l6Reserves*probOFa_oda"] = data4["IV_reserves_OFa_oda_1_ln"].shift(6)
    data4["l6factor1*probOFa_oda"] = data4["IV_factor1_OFa_oda_1_ln"].shift(6)
# +6
    data4["l6OFa_all"] = data4['OFa_all_con_ln'].shift(6)
    data4["l7Reserves*probOFa_all"] = data4["IV_reserves_OFa_all_1_ln"].shift(7)
    data4["l7factor1*probOFa_all"] = data4["IV_factor1_OFa_all_1_ln"].shift(7)
    data4["l6OFa_oofv"] = data4['OFa_oofv_con_ln'].shift(6) #oofv
    data4["l7Reserves*probOFa_oofv"] = data4["IV_reserves_OFa_oofv_1_ln"].shift(7)
    data4["l7factor1*probOFa_oofv"] = data4["IV_factor1_OFa_oofv_1_ln"].shift(7)
    data4["l6OFa_oda"] = data4['OFa_oda_con_ln'].shift(6) #oda
    data4["l7Reserves*probOFa_oda"] = data4["IV_reserves_OFa_oda_1_ln"].shift(7)
    data4["l7factor1*probOFa_oda"] = data4["IV_factor1_OFa_oda_1_ln"].shift(7)
    
    # supress special type of warnings
    pd.options.mode.chained_assignment = None  
    

    # Set observations for countries to NaN according to Appendix C1 of the paper
    data4.loc[data4.countryname == "Antigua and Barbuda"] = np.nan #only one observation
    data4.loc[data4.countryname == "China"] = np.nan               #not interested in impact on China
    data4.loc[data4.countryname == "Barbados"] = np.nan            #only one observation
    
    # apply final cutoff and cleaning up
    data4 = data4[data4.year >= "1997-01-01"].copy()
    data4.replace([np.inf, -np.inf], np.nan, inplace=True)
    data4 = data4.dropna(0, subset = ["l1population_ln", "growth_pc"])
    #(0, subset = ["l1population_ln"])#, "l2FDI_China_ln", "l2Exports_ln"])
    
    
    

    if multiindex == True:
        data4 = data4.set_index(["countryname", "year"])
    
    return(data4)
