import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as cnf
from itertools import combinations
from sklearn.model_selection import GridSearchCV, cross_validate

def check_df(dataframe, head=5,non_numeric=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### DESCRIBE #####################")
    print(dataframe.describe())
    
    for col in dataframe.columns:
        if dataframe[col].isna().sum() <= 0:
            if dataframe[col].nunique() > 20:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()}\n')
            else:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()} --- unique: {dataframe[col].unique()}\n')
        else:
            if dataframe[col].nunique() > 20:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()} --- nan: {dataframe[col].isna().sum()}\n')
            else:
                print("##################### COLUMN #####################")
                print(f'{col} --- nunique: {dataframe[col].nunique()} --- unique: {dataframe[col].unique()} --- nan: {dataframe[col].isna().sum()}\n')
    
    if non_numeric:
        print("##################### Quantiles #####################")
        print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    
def num_summary(dataframe, numerical_col, plot=False):
    """
        Numerik kolonlar input olarak verilmelidir.
        Sadece ekrana cikti veren herhangi bir degeri return etmeyen bir fonksiyondur.
        For dongusuyle calistiginda grafiklerde bozulma olmamaktadir.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
      
def plot_distributions(dataframe, columns,kde=False, log_transform=False, label_angle=0, 
                       figsize = (8,3) , order_cats= False, target_pie=False, alert=False,target=cnf.target): 

    if alert == True:
        pie_palette = cnf.alert_palette
    else:
        pie_palette = cnf.sequential_palette
        
    if target_pie == True:
#         colors = ['#ff6666', '#468499', '#ff7f50', '#ffdab9', 
#           '#00ced1', '#ffff66','#088da5','#daa520',
#           '#794044','#a0db8e','#b4eeb4','#c0d6e4','#065535','#d3ffce']
# fig1, ax1 = plt.subplots(figsize=(14,7))

# ax1.pie(data.enflasyon,labels=data.Tarih,colors=colors, autopct='%1.1f%%');
        ax = dataframe[columns].value_counts().plot.pie(autopct='%1.1f%%',
                                              textprops={'fontsize':10},
                                              colors=cnf.muted_palette
                                              ).set_title(f"{target} Distribution")
        plt.ylabel('')
        plt.show()

    else:
        for col in columns:
            if log_transform == True:
                x = np.log10(dataframe[col])
                title = f'{col} - Log Transformed'
            else:
                x = dataframe[col]
                title = f'{col}'
            
            if order_cats == True:
                
                print(pd.DataFrame({col: dataframe[col].value_counts(),
                            "Ratio": 100 * dataframe[col].value_counts() / len(dataframe)}))
            
                print("##########################################")
                
                print(f"NA in {col} : {dataframe[col].isnull().sum()}")
                
                print("##########################################")

                labels = dataframe[col].value_counts(ascending=False).index
                values = dataframe[col].value_counts(ascending=False).values
                
                plt.subplots(figsize=figsize)
                plt.tight_layout()
                plt.xticks(rotation=label_angle)
                sns.barplot(x=labels,
                            y=values,
                            palette = cnf.sequential_palette)
                        
            else:   
            
                quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
                print(dataframe[col].describe(quantiles).T)

                plt.subplots(figsize=figsize)
                plt.tight_layout()
                plt.xticks(rotation=label_angle)
                sns.histplot(x,
                        bins=50,
                        kde=kde,
                        color = cnf.sequential_palette[0])

    
            plt.title(title)
            plt.show()

def numcols_target_corr(dataframe, num_cols,target = cnf.target):
    numvar_combinations = list(combinations(num_cols, 2))
    
    for item in numvar_combinations:
        
        plt.subplots(figsize=(8,4))
        sns.scatterplot(x=dataframe[item[0]], 
                        y=dataframe[item[1]],
                        hue=dataframe[target],
                        palette=cnf.bright_palette
                       ).set_title(f'{item[0]}   &   {item[1]}')
        plt.grid(True)
        plt.show()            
            
def numeric_variables_boxplot(df, num_cols, target=None, alert=False):
    
    if alert == True:
        palette = cnf.alert_palette
    else:
        palette = cnf.bright_palette
        
    if target == None:
        
        fig, [ax1,ax2,ax3,ax4] = plt.subplots(1,4, figsize=(7,3))

        for col, ax, i in zip(num_cols, [ax1,ax2,ax3,ax4], range(4)):
            sns.boxplot(df[col], 
                        color=palette[i], 
                        ax=ax
                        ).set_title(col)
            
        for ax in [ax1,ax2,ax3,ax4]:
            ax.set_xticklabels([])
    else:
        for col in num_cols:
            plt.subplots(figsize=(7,3))
            sns.boxplot(x=df[target], 
                                y=df[col],
                                hue=df[target],
                                dodge=False, 
                                fliersize=3,
                                linewidth=0.7,
                                palette=palette)
            plt.title(col)
            plt.xlabel('')
            plt.ylabel('')
            plt.xticks(rotation=45)
            plt.legend('',frameon=False)

    plt.tight_layout()
    plt.show()
    
def plot_categorical_data(dataframe, x, hue, title='', label_angle=0):
    """
    Kategorik veri görselleştirmesi için alt grafikleri çizen bir fonksiyon. 
    """
    # Alt grafikleri yan yana düzenleme
    fig, ax = plt.subplots(1, figsize=(8, 3))

    # Grafik 1
    sns.countplot(data=dataframe, x=x, hue=hue, ax=ax, palette='husl')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)
    ax.legend(prop={'size': 10})

    # Grafikleri göster
    plt.tight_layout()
    plt.xticks(rotation=label_angle)
    plt.show(block=True)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
