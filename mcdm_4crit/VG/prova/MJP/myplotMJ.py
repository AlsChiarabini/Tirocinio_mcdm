import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def mybarplot(data, portfolio='portfolio', data_col='data', factor='factor', nameplotx = 'portfolios', save='Y', file_format='png', save_as='data.png',  palette="tab20"):
    sns.set_style("whitegrid")
    plt.figure(figsize=(8.2, 5))
    
    # Set the color palette
    if palette == 'Paired':
        sns.set_palette(sns.color_palette(palette,12).as_hex()[:-2])
    else:
        sns.set_palette(palette,12)
    
    ax = sns.barplot(x=portfolio, y=data_col, hue=factor, data=data)
    
    # Get the current labels
    labels = [item.get_text() for item in ax.get_xticklabels()]
    
    # Rename 'long_short' to 'High-Low'
    labels = ['High-Low' if label == 'long_short' else label for label in labels]
    
    # Set the new labels
    ax.set_xticklabels(labels)
    ax.set_xlabel(nameplotx)
    
    # Get the current handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    sns.despine()
    # Find the index of 'dolvol6' and 'ill6' in labels and replace them
    for i, label in enumerate(labels):
        if label == 'dolvol6':
            labels[i] = r'dolvol$_{6M}$'
        if label == 'ill6':
            labels[i] = r'ill$_{6M}$'
    
    # Set the legend again
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Save the plot in the specified format if save is 'Y'
    if save.upper() == 'Y':
        if file_format == 'both':
            plt.savefig(save_as, bbox_inches='tight')
            plt.savefig(save_as.replace('.png', '.eps'), format='eps', bbox_inches='tight')
        elif file_format == 'eps':
            plt.savefig(save_as.replace('.png', '.eps'), format='eps', bbox_inches='tight')
        else:
            plt.savefig(save_as, bbox_inches='tight')
    plt.show()



def cumret_plot(data_returns, factors, signs, plot_type='long_short', single_factor ='size', num_port =10,  save='Y', file_format='png', suffix = '', palette='tab20'):
    #plt.figure(figsize=(6.5, 4.5))
    
    # Set the color palette
    if palette == 'Paired':
        paired_palette = sns.color_palette(palette, 12)
        custom_palette = paired_palette[:10] + [paired_palette[-1]]  # Skip the penultimate color
        sns.set_palette(custom_palette)
    else:
        sns.set_palette(sns.color_palette(palette, 11))
    
    # Plot each factor
    if 'wmj' in data_returns:
        modemj = 'wmj'
        tit = 'VW'
    elif 'mj' in data_returns:
        modemj = 'mj'
        tit = 'EW'
        
    if plot_type == 'long_short': 
        # plot the cumulative returns of the long short strategy using all the factors
        for key in factors+[modemj]:
            plt.plot(((1+data_returns[key])['long_short'].cumprod()-1),label=key)
        title = f'Hign-Low {tit} portfolios'
        save_as = f'figure/{tit}_HignLow_port{suffix}'
    elif plot_type == 'portfolio':
        # plot the cumulative returns of a specific portfolio using all the factors
        for key in factors+[modemj]:
            plt.plot(((1+data_returns[key])[f'port{num_port}'].cumprod()-1),label=key)
        title = f'{tit} portfolio {num_port}'
        save_as = f'figure/{tit}_port{num_port}{suffix}'
    elif plot_type == 'single_factor':
        # plot the cumulative returns of a specific characteristic using all the portfolios
        ((1+data_returns[single_factor])[[f'port{i}' for i in range(1,num_port+1)]].cumprod()-1).plot()
        title = f'{tit} {single_factor} portfolios'
        save_as = f'figure/{tit}_{single_factor}port{suffix}'
    elif plot_type == 'winners':
        for key, sign in zip(factors, signs):
            if sign == -1:
                plt.plot(((1+data_returns[key])[f'port{1}'].cumprod()-1), label=key)
            else:
                plt.plot(((1+data_returns[key])[f'port{num_port}'].cumprod()-1), label=key)
        plt.plot(((1+data_returns[modemj])[f'port{num_port}'].cumprod()-1), label=modemj)      
        title = f'{tit} winner portfolios'
        save_as = f'figure/{tit}_portwinner{suffix}'
            
    
    # Remove the grid
    plt.grid(False)
    
    # Add legend, title, and labels
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    
    # Get the current handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Find the index of 'dolvol6' and 'ill6' in labels and replace them
    for i, label in enumerate(labels):
        if label == 'dolvol6':
            labels[i] = r'dolvol$_{6M}$'
        if label == 'ill6':
            labels[i] = r'ill$_{6M}$'
    
    # Set the legend again
    if plot_type == 'long_short':
        plt.legend(handles, labels, bbox_to_anchor=(1,1), loc="upper left")
    else:
        plt.legend(handles, labels)
        
    # Save the plot in the specified format if save is 'Y'
    if save.upper() == 'Y':
        if file_format == 'both':
            plt.savefig(save_as, bbox_inches='tight')
            plt.savefig(save_as.replace('.png', '.eps'), format='eps', bbox_inches='tight')
        elif file_format == 'eps':
            plt.savefig(save_as.replace('.png', '.eps'), format='eps', bbox_inches='tight')
        else:
            plt.savefig(save_as, bbox_inches='tight')
    plt.show()



def cumret_plotv2(data_returns, factors, signs, plot_type='long_short', single_factor ='size', num_port =10,  save='Y', file_format='png', suffix = '', palette='tab20', figsize=(9,6)):
    #plt.figure(figsize=(6.5, 4.5))
    
    # Set the color palette
    if palette == 'Paired':
        paired_palette = sns.color_palette(palette, 12)
        custom_palette = paired_palette[:10] + [paired_palette[-1]]  # Skip the penultimate color
        sns.set_palette(custom_palette)
    else:
        sns.set_palette(sns.color_palette(palette, 11))
    
    plt.figure(figsize=figsize)
    # Plot each factor
    if 'wmj' in data_returns:
        modemj = 'wmj'
        tit = 'VW'
    elif 'mj' in data_returns:
        modemj = 'mj'
        tit = 'EW'
        
    if plot_type == 'long_short': 
        # plot the cumulative returns of the long short strategy using all the factors
        for key in factors+[modemj]:
            plt.plot(((1+data_returns[key])['long_short'].cumprod()-1),label=key)
        save_as = f'{tit}_HignLow_port{suffix}'
    elif plot_type == 'portfolio':
        # plot the cumulative returns of a specific portfolio using all the factors
        for key in factors+[modemj]:
            plt.plot(((1+data_returns[key])[f'port{num_port}'].cumprod()-1),label=key)
        save_as = f'{tit}_port{num_port}{suffix}'
    elif plot_type == 'single_factor':
        # plot the cumulative returns of a specific characteristic using all the portfolios
        ((1+data_returns[single_factor])[[f'port{i}' for i in range(1,num_port+1)]].cumprod()-1).plot(figsize=figsize)
        save_as = f'{tit}_{single_factor}port{suffix}'
    elif plot_type == 'winners':
        for key, sign in zip(factors, signs):
            if sign == -1:
                plt.plot(((1+data_returns[key])[f'port{1}'].cumprod()-1), label=key)
            else:
                plt.plot(((1+data_returns[key])[f'port{num_port}'].cumprod()-1), label=key)
        plt.plot(((1+data_returns[modemj])[f'port{num_port}'].cumprod()-1), label=modemj)      
        save_as = f'{tit}_portwinner{suffix}'
            
    
    # Remove the grid
    plt.grid(False)
    
    # Add legend, title, and labels
    #plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Excess Returns')
    
    # Get the current handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Find the index of 'dolvol6' and 'ill6' in labels and replace them
    for i, label in enumerate(labels):
        if label == 'dolvol6':
            labels[i] = r'dolvol$_{6M}$'
        if label == 'ill6':
            labels[i] = r'ill$_{6M}$'
    
    # Set the legend again
    if plot_type == 'long_short':
        plt.legend(handles, labels, bbox_to_anchor=(1,1), loc="upper left")
    else:
        plt.legend(handles, labels)
        
    # Save the plot in the specified format if save is 'Y'
    if save.upper() == 'Y':
        if file_format == 'both':
            plt.savefig(save_as, bbox_inches='tight')
            plt.savefig(save_as.replace('.png', '.eps'), format='eps', bbox_inches='tight')
        elif file_format == 'eps':
            plt.savefig(save_as.replace('.png', '.eps'), format='eps', bbox_inches='tight')
        else:
            plt.savefig(save_as+'.pdf', bbox_inches='tight')
    plt.show()
    
    
    
def barplot_variations(df,port=10,
                       fix_mode=True,
                       mode='vote',
                       estimator=np.mean,
                       ci=85,
                       capsize=.2,
                       figsize=(6.2,3),
                       save=False,
                       prefix='',
                       sup_xlim=None,
                       ax_legend=0):
    if fix_mode:
        df=df.loc[df['mode']==mode]
        pre_title=f'{mode} rolling, '

    mask1=f"weighting == 'even'"
    mask2=f"weighting == 'weighted'"
    masks=[mask1,mask2]
    dict_title={
        mask1:'EW',
        mask2:'VW'
    }
    
    if port=='long_short':
        add_title=port+' portfolio, '
    else:
        add_title=f'portfolio {str(port)}, '

    if isinstance(port,int):
        port10=df.loc[df['portfolio']==f'port{port}']
    else: 
        port10=df.loc[df['portfolio']==port]

    port10['method']=port10['method'] + 'M'
    
    _, axs = plt.subplots(1, 2, layout='constrained', sharex=True, sharey=True, figsize=figsize)
    if fix_mode:
        for i,mask in enumerate(masks):
            if not ci:
                sns.barplot(x = 'returns', y = 'method', hue = 'outlier', 
                            data = port10.query(mask),palette = 'tab20', ax=axs[i],
                            estimator=estimator, capsize=capsize,errwidth=1)
            else:
                sns.barplot(x = 'returns', y = 'method', hue = 'outlier', 
                            data = port10.query(mask),palette = 'tab20', ax=axs[i],
                            estimator=estimator,errorbar=('ci', ci), capsize=capsize,errwidth=1)
            axs[i].set_title(add_title + pre_title + dict_title[mask])    
        if ax_legend==0:
            axs[0].legend(framealpha=0.5)
            axs[0].set_ylabel('mj rolling window')
            #axs[1].legend(loc=[0.8, 0.78],framealpha=0.5)
            axs[1].get_legend().set_visible(False)
            axs[1].set_ylabel('')
        else:
            axs[1].legend(framealpha=0.5)
            axs[0].set_ylabel('mj rolling window')
            #axs[1].legend(loc=[0.8, 0.78],framealpha=0.5)
            axs[0].get_legend().set_visible(False)
            axs[1].set_ylabel('')
        if sup_xlim:
            axs[0].set_xlim([0,sup_xlim])
            axs[1].set_xlim([0,sup_xlim])

    else:
        for i,mask in enumerate(masks):
            if not ci:
                sns.barplot(x = 'returns', y = 'mode', hue = 'outlier', 
                            data = port10.query(mask),palette = 'tab20', ax=axs[i],
                            estimator=estimator, capsize=capsize,errwidth=1)
            else:
                sns.barplot(x = 'returns', y = 'mode', hue = 'outlier', 
                            data = port10.query(mask),palette = 'tab20', ax=axs[i],
                            estimator=estimator,errorbar=('ci', ci), capsize=capsize,errwidth=1)
            axs[i].set_title(add_title + dict_title[mask])  
        if ax_legend==0:
            axs[0].legend(framealpha=0.5)  
            axs[0].set_ylabel('rolling window approach')
            #axs[1].legend(loc=[0.8, 0.78],framealpha=0.5)
            axs[1].get_legend().set_visible(False)
            axs[1].set_ylabel('')
        else:
            axs[1].legend(framealpha=0.5)  
            axs[0].set_ylabel('rolling window approach')
            #axs[1].legend(loc=[0.8, 0.78],framealpha=0.5)
            axs[0].get_legend().set_visible(False)
            axs[1].set_ylabel('')
        if sup_xlim:
            axs[0].set_xlim([0,sup_xlim])
            axs[1].set_xlim([0,sup_xlim])
    plt.tight_layout()
    if save:
        if fix_mode:
            plt.savefig(f'./{prefix}_port_{port}_{mode}_ewvw_out.pdf')
        else:
            plt.savefig(f'./{prefix}_port_{port}_wmj_averaged_ewvw_out.pdf')
    plt.show()
        
        
def barplot_robustness(df,
                       port=10,
                       estimator=np.mean, 
                       capsize=.2,
                       figsize=(6.2,3),
                       save=False,
                       prefix=''):
    
    mask1=f"weighting == 'even'"
    mask2=f"weighting == 'weighted'"
    masks=[mask1,mask2]
    dict_title={
        mask1:'EW',
        mask2:'VW'
    }
    
    if port=='long_short':
        add_title=port+' portfolio, '
    else:
        add_title=f'portfolio {str(port)}, '
        
    portfolio=df.loc[df['portfolio']==port]

    _, axs = plt.subplots(1, 2, layout='constrained', sharex=True, sharey=True, figsize=figsize)
    
    for i,mask in enumerate(masks):
        sns.barplot(x = 'returns', y = 'method', hue = 'outlier', 
                    data = portfolio.query(mask),palette = 'tab20', ax=axs[i],
                    estimator=estimator, errorbar=custom_error, capsize=capsize,
                    errwidth=1)
        axs[i].set_title(add_title + dict_title[mask])    
    axs[0].get_legend().set_visible(False)
    axs[0].set_ylabel('Number of voters')
    #axs[1].legend(loc=[0.8, 0.78],framealpha=0.5)
    axs[1].legend(framealpha=0.5)
    axs[1].set_ylabel('')
    plt.tight_layout()
    if save:
        plt.savefig(f'./{prefix}_port_{port}_robustness.pdf')
    plt.show()
    
    
def custom_error(data):
    return (np.quantile(data,0.05),np.quantile(data,0.95),)