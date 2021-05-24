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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    