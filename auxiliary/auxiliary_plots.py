"""This module contains auxiliary functions for plotting which are used in the main notebook."""

import numpy as np
import pandas as pd
import pandas.io.formats.style
import seaborn as sns
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm_api
import matplotlib as plt
import geopandas 
from IPython.display import HTML


def worldplot(data):
    
    """ Function to plot a custom colored worldmap. Therefore we have to input a dataframe that contains the column on which 
    conditionally we want o color the worldmap

        Args:
        -------
            data = pd.dataframe wich contains column of interes

        Returns:
        ---------
            plot
    """
    
    plt.rcParams['font.size'] = 18
    world_df= geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'));

    world_df = world_df[world_df["iso_a3"].isin(data["recipient_iso3"])];

    #world_2df.["OFa_all_con"] = np.nan;
    #world_2df.sort_values(by="iso_a3").head()
    for i in world_df.index:
        for j in data.index:
            if world_df.loc[i,"iso_a3"] == data.loc[j,"recipient_iso3"]:
                world_df.loc[i,"OFa_all_con"] = data.loc[j, "OFa_all_con"];


    fig, ax = plt.pyplot.subplots(1,1, figsize=(22,14))
    ax.axis('off')
    fig.suptitle('Chinese Development Finance', fontsize=25)
    
    world_df.plot(column='OFa_all_con', ax = ax, legend=True, legend_kwds={"label":"\n Chinese Development Finance in $10 bln.",
                                                                         "orientation": "horizontal"}, 
                                                              missing_kwds={"color": "lightgrey",
                                                                            "edgecolor": "red",
                                                                            "hatch": "///",
                                                                            "label": "Missing values"});
###
    
def worldplot_2(data, cc, pc):
    
    """ Function to plot a custom colored worldmap with help of a standart GeoPandas dataframe.
    Therefore we have to input a dataframe that contains the iso3 code of countries (checkcol). Furthermore you need
    to specify the column of the input data that you want to display on the worldmap

        Args:
        -------
            data = pd.dataframe wich contains column of interest
            checkcol = columnnumber of iso_3 code of input df
            plotcol = the columnnumber of the column that we want to plot

        Returns:
        ---------
            The return is a formated plot
    """
   # define the columns of input
   # cc = data.columns[checkcol]
    #pc = data.columns[plotcol]
    
    plt.rcParams['font.size'] = 18
    # generate standart geopandas dataframe
    world_df = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'));
    #check indicies of the input dataframe and modify standart geopandas df
    world_df = world_df[world_df["iso_a3"].isin(data[cc])];

    #world_2df.["OFa_all_con"] = np.nan;
    #world_2df.sort_values(by="iso_a3").head()
    for i in world_df.index:
        for j in data.index:
            if world_df.loc[i,"iso_a3"] == data.loc[j, cc]:
                world_df.loc[i,pc] = data.loc[j, pc];

    fig, ax = plt.pyplot.subplots(1,1, figsize=(22,12))
    ax.axis('off')
    fig.suptitle('Chinese Development Finance', fontsize=25)
    
    world_df.plot(column=pc, ax = ax, legend=True, legend_kwds={"label":"\n Chinese Development Finance in $10 bln",
                                                                         "orientation": "horizontal"}, 
                                                              missing_kwds={"color": "lightgrey",
                                                                            "edgecolor": "red",
                                                                            "hatch": "///",
                                                                            "label": "Missing values"});

###

def flow_class_plot(data):    
    sns.set_theme(style="whitegrid")
    f, axs = plt.pyplot.subplots(1,2,figsize=(15,15))
    plt.pyplot.subplots_adjust(wspace=0.5)

    plotting = data.flow_class.value_counts(1)
    plt.pyplot.subplot(121)
    ax = sns.barplot(x=plotting.index, y=plotting.values)
    ax.set_ylabel("share")
    ax.set_title("Share of flow_class");


    plotting2 = data.groupby("flow_class").usd_current.sum()
    plt.pyplot.subplot(122)
    ax = sns.barplot(x=plotting2.index, y=(plotting2.values/1e6))
    ax.set_ylabel("Amount in million USD")
    ax.set_title("Share of flow_class");
    
    plt.pyplot.plot()

    df = pd.DataFrame([["ODA", plotting[0], plotting2[0]/1e6],
                      ["OOF", plotting[1], plotting[1]/1e6],
                      ["Vague",plotting[2], plotting2[2]/1e6]], columns = ["flow_class", "Share", "Amount in USD"])
    
    #print(f"Number of projects:\nODA-like:{plotting[0]*100}%, \nOOF-like:{plotting[1]*100}%, \nVague OF:{plotting[2]*100}%")
   # print(f"ODA-like:{plotting2[0]/1e6:.2f}, \nOOF-like:{plotting2[1]/1e6:.2f}, \nVague OF:{plotting2[2]/1e6:.2f}")
    #print((plotting2[0]/1e6)/(plotting2.values.sum()/1e6))
    
    return df
    
###

def year_plot(data):
    sns.set_theme(style="whitegrid")
    year = np.unique(data.year)
    total_projects_year = data.year.value_counts().sort_index()

    ax = sns.barplot(x=year, y=total_projects_year, color= "blue")
    ax.set_ylabel("number of projects")
    ax.set_xticklabels(["'00", "'01", "'02","'03","'04","'05","'06","'07","'08","'09","'10","'11","'12","'13","'14"])
    ax.set_title("number of projects per year");

###

def sectoral_plot(data):

    sns.set_theme(style="whitegrid")
    sectoral_analysis_df = data.crs_sector_name.value_counts(1).sort_index().to_frame("project_share")
    sectoral_analysis_df["in_USD"] = data.groupby("crs_sector_name").usd_current.sum()
    sectoral_analysis_df = sectoral_analysis_df.sort_values(by="in_USD", ascending=False)

    # plotting
    f, axs = plt.pyplot.subplots(2,1,figsize=(15,15))
    plt.pyplot.subplot(211)
    ax = sns.barplot(y=sectoral_analysis_df.index, x=sectoral_analysis_df.in_USD, color = "darkblue")
    ax.set_title("Value per sector");

    plt.pyplot.subplot(212)
    ax = sns.barplot(y=sectoral_analysis_df.index, x=sectoral_analysis_df.project_share, color = "lightblue")
    ax.set_title("Sare of projects per sector");
    
    # Share of health, education and governance
    share_HEG = ((sectoral_analysis_df.loc["Health", "in_USD"] + sectoral_analysis_df.loc["Education", "in_USD"]
                 + sectoral_analysis_df.loc["Government and Civil Society", "in_USD"]) / sectoral_analysis_df["in_USD"].sum()) / 1e6
    
    # Share of energy, transport, industry
    share_ETI = ((sectoral_analysis_df.loc["Energy Generation and Supply", "in_USD"] 
                  + sectoral_analysis_df.loc["Industry, Mining, Construction", "in_USD"]
                  + sectoral_analysis_df.loc["Transport and Storage", "in_USD"]) / sectoral_analysis_df["in_USD"].sum()) / 1e6
    
    print(share_HEG)
    


