"""This module contains auxiliary functions for plotting which are used in the main notebook."""

import numpy as np
import pandas as pd
import pandas.io.formats.style
import seaborn as sns
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm_api
import matplotlib.pyplot as plt
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


    fig, ax = plt.subplots(1,1, figsize=(22,14))
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
    
    """ Function to plot a custom colored worldmap with help of a standart GeoPandas dataframe. I used the iso3 number of the countries
    in order to clearly identify the countries and assign the choosen value (financial amount or project count) to the 
    specific country
    For plotting, we have to input a dataframe that contains the iso3 code of countries (cc). Furthermore you need
    to specify the column of the input data that you want to display on the worldmap (pc)


        Args:
        -------
            data = pd.dataframe wich contains column of interest
            cc = columnnumber of country of input df
            pc = the columnnumber of the column that we want to plot

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
                try:
                    world_df.loc[i,pc] = data.loc[j, pc];
                except:    
                    print("\nError! Invalid Input. Example for input: OFa_all_con")
                    return
                   

    fig, ax = plt.subplots(1,1, figsize=(22,12))
    ax.axis('off')
    fig.suptitle('Chinese Development Finance', fontsize=25)
    
    if pc == "OFa_all_con":
        world_df.plot(column=pc, ax = ax, legend=True, legend_kwds={"label":"\n Chinese Development Finance in $10 bln",
                                                                         "orientation": "horizontal"}, 
                                                              missing_kwds={"color": "lightgrey",
                                                                            "edgecolor": "red",
                                                                            "hatch": "///",
                                                                            "label": "Missing values"});
    else:
        world_df.plot(column=pc, ax = ax, legend=True, legend_kwds={"label":"\n Chinese Development Finance project count",###ADDDDJUST!!!!!
                                                                         "orientation": "horizontal"}, 
                                                              missing_kwds={"color": "lightgrey",
                                                                            "edgecolor": "red",
                                                                            "hatch": "///",
                                                                            "label": "Missing values"});
 
###

def flow_class_plot(data):    
    sns.set_theme(style="whitegrid")
    
    plt.subplots_adjust(wspace=0.5)

    plotting = data.flow_class.value_counts(1)
    plt.subplot(121)
    ax = sns.barplot(x=plotting.index, y=plotting.values)
    ax.set_ylabel("share")
    ax.set_title("Share of flow_class");


    plotting2 = data.groupby("flow_class").usd_defl.sum()
    plt.subplot(122)
    ax = sns.barplot(x=plotting2.index, y=(plotting2.values/1e6))
    ax.set_ylabel("Amount in million USD")
    ax.set_title("Share of flow_class");
    
    plt.plot()

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
    sectoral_analysis_df["in_USD"] = data.groupby("crs_sector_name").usd_defl.sum()
    sectoral_analysis_df = sectoral_analysis_df.sort_values(by="in_USD", ascending=False)

    # plotting
    f, axs = plt.subplots(2,1,figsize=(15,15))
    plt.subplot(211)
    ax = sns.barplot(y=sectoral_analysis_df.index, x=sectoral_analysis_df.in_USD, color = "darkblue")
    ax.set_title("Value per sector");

    plt.subplot(212)
    ax = sns.barplot(y=sectoral_analysis_df.index, x=sectoral_analysis_df.project_share, color = "lightblue")
    ax.set_title("Sare of projects per sector");
    
    # Share of health, education and governance
    share_HEG = ((sectoral_analysis_df.loc["Health", "in_USD"] + sectoral_analysis_df.loc["Education", "in_USD"]
                 + sectoral_analysis_df.loc["Government and Civil Society", "in_USD"]) / sectoral_analysis_df["in_USD"].sum())
    
    # Share of energy, transport, industry
    share_ETI = ((sectoral_analysis_df.loc["Energy Generation and Supply", "in_USD"] 
                  + sectoral_analysis_df.loc["Industry, Mining, Construction", "in_USD"]
                  + sectoral_analysis_df.loc["Transport and Storage", "in_USD"]) / sectoral_analysis_df["in_USD"].sum()) 
    
    print(f"All projects of the health-, education and governance sector account for {share_HEG*100:.2f}%,\nwhereas the energy-, transportation and industry/mining sector accounts for {share_ETI*100:.2f}%")

###
def quali_descriptive_plots(data, liste):
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    # Use the axes for plotting
    axes[0,0].set_title(liste[0])
    sns.violinplot(x=liste[0], y="OFn_all", data=data, ax=axes[0,0], inner = "quartiles");

    axes[0,1].set_title(liste[0])
    ax = sns.violinplot(x=liste[0], y="OFa_all_con", data=data, ax=axes[0,1], inner = "quartiles")

    axes[1,0].set_title(liste[1])
    sns.violinplot(x=liste[1], y="OFn_all", data=data,ax=axes[1,0], inner = "quartiles");

    axes[1,1].set_title(liste[1])
    ax = sns.violinplot(x=liste[1], y="OFa_all_con", data=data,ax=axes[1,1], inner = "quartiles");

    axes[2,0].set_title(liste[2])
    sns.violinplot(x=liste[2], y="OFn_all", data=data,ax=axes[2,0], inner = "quartiles");

    axes[2,1].set_title(liste[2])
    ax = sns.violinplot(x=liste[2], y="OFa_all_con", data=data,ax=axes[2,1], inner = "quartiles");

    plt.tight_layout(pad=2.5);


###
def quanti_descriptive_plots(data, liste):
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    # Use the axes for plotting
    axes[0,0].set_title(liste[0])
    sns.scatterplot(x=liste[0], y="OFn_all", data=data, ax=axes[0,0], hue = "A_D99petroleum", style ="A_Ltaiwanr")

    axes[0,1].set_title(liste[0])
    ax = sns.scatterplot(x=liste[0], y="OFa_all_con", data=data, ax=axes[0,1], hue = "A_D99petroleum", style ="A_Ltaiwanr")

    axes[1,0].set_title(liste[1])
    sns.scatterplot(x=liste[1], y="OFn_all", data=data,ax=axes[1,0], hue = "A_D99petroleum", style ="A_Ltaiwanr");

    axes[1,1].set_title(liste[1])
    ax = sns.scatterplot(x=liste[1], y="OFa_all_con", data=data,ax=axes[1,1], hue = "A_D99petroleum", style ="A_Ltaiwanr");

    axes[2,0].set_title(liste[2])
    sns.scatterplot(x=liste[2], y="OFn_all", data=data,ax=axes[2,0], hue = "A_D99petroleum", style ="A_Ltaiwanr");

    axes[2,1].set_title(liste[2])
    ax = sns.scatterplot(x=liste[2], y="OFa_all_con", data=data,ax=axes[2,1], hue = "A_D99petroleum", style ="A_Ltaiwanr");
    
    axes[3,0].set_title(liste[3])
    sns.scatterplot(x=liste[2], y="OFn_all", data=data,ax=axes[3,0], hue = "A_D99petroleum", style ="A_Ltaiwanr");

    axes[3,1].set_title(liste[3])
    ax = sns.scatterplot(x=liste[2], y="OFa_all_con", data=data,ax=axes[3,1], hue = "A_D99petroleum", style ="A_Ltaiwanr");

    plt.tight_layout(pad=2.5);
