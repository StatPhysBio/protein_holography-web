
import numpy as np

# import Bio.PDB as pdb
# import h5py
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import auc

from protein_holography_web.utils.protein_naming import *

markersize=4
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# TICK_SIZE = 32
# plt.rc('xtick', labelsize=TICK_SIZE)    
# plt.rc('ytick', labelsize=TICK_SIZE) 
from matplotlib.gridspec import GridSpec
plt.rcParams['font.size'] = 6.
plt.rcParams['font.family'] = "Arial"

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

def x_cell(x,y, ax=None, **kwargs):
    ax.scatter(x, y, marker='x', **kwargs)
    ax = ax or plt.gca()
    
import matplotlib.colors as mcolors
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def keep_center_colormap(vmin, vmax, center=0):
    vmin = vmin - center
    vmax = vmax - center
    dv = max(-vmin, vmax) * 2
    N = int(256 * dv / (vmax-vmin))
    RdBu = cm.get_cmap('RdBu', N)
    newcolors = RdBu(np.linspace(0, 1, N))
    beg = int((dv / 2 + vmin) * N / dv)
    end = N - int((dv / 2 - vmax) * N / dv)
    newmap = mcolors.ListedColormap(newcolors[beg:end])
    return newmap


def plot_heatmap(df,hm_value,hm_title,ax,vmin=None,vmax=None,vcenter=0.,cax=None):
        if vmin == None:
            vmin,vmax = (
                np.min(
                    df[[hm_value.format(aa_to_one_letter[ind_to_aa_ward[x]])  
                   for x in range(20) ]].to_numpy()
                ),
                np.max(
                    df[[hm_value.format(aa_to_one_letter[ind_to_aa_ward[x]])  
                   for x in range(20) ]].to_numpy()
                )
            )

        newmap = keep_center_colormap(vmin,vmax)
        hm_im = ax.imshow(
            df[[hm_value.format(aa_to_one_letter[ind_to_aa_ward[x]])  
                   for x in range(20) ]],
            cmap='RdBu',
#             vmin=vmin,vmax=vmax
            norm=mcolors.TwoSlopeNorm(vmin=vmin,vmax=vmax,vcenter=vcenter)
        )
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(labels=df['mutant'])
        ax.set_xticks(range(20))
        ax.set_xticklabels(labels=[aa_to_one_letter[ind_to_aa_ward[x]] for x in range(20)])
        ax.set_title(hm_title)

        for j,row in df.iterrows():
            wt_idx = aa_to_ind_ward[one_letter_to_aa[row['wt']]]
            mut_idx = aa_to_ind_ward[one_letter_to_aa[row['mut']]]
            x_cell(
                wt_idx,
                j,
                color='black',
                ax=ax,
                s=1,
                linewidth=1,
            )
            mut_color = 'black'
            highlight_cell(
                mut_idx,
                j,
                ax=ax,
                color=mut_color,
                linestyle=':',
                linewidth=0.5
            )
        ax.plot(
            [-0.5,19.5],[19.5,19.5],
            color='k',
            linestyle='--',
            linewidth=0.5
        )
        if cax==None:
            pos = ax.get_position()
            fig = ax.get_figure()
            cax = fig.add_axes(
                [pos.x0-pos.width*0.1,pos.y0-pos.height*0.2,
                 pos.width,pos.height*0.02]
            )
        plt.colorbar(
            hm_im,
            cax=cax,
            orientation='horizontal',
            shrink=0.3,
        )
            
def plot_heatmaps(df,hm_values,hm_titles,filename=None,vcenter=0.):
    n = len(hm_values)
    fig = plt.figure(figsize=(4*n,4),dpi=300,constrained_layout=True)
    gs = GridSpec(
        1,n,
        figure=fig,
        width_ratios=[20]*n,
         #height_ratios=[20,20],
#         wspace=0.5,
#         hspace=0.5
    )
    hm_axes = [fig.add_subplot(gs[:,i]) for i in range(n)]
    
    plotted_fields = np.concatenate(
        [
            *[[x for x in df.columns if y.strip('{}') in x] for y in hm_values]
        ]
    )
    all_data = df[
        plotted_fields
    ]
    vmin,vmax = np.min(all_data.to_numpy()),np.max(all_data.to_numpy())
#     print(vmin,vmax)
#     print(all_data)
    for i,hm_value in enumerate(hm_values):
        plot_heatmap(df,hm_value,hm_titles[i],hm_axes[i],vmin,vmax,vcenter=vcenter,)
#         divider = make_axes_locatable(hm_axes[i])
#         cax = divider.append_axes("bottom", size="2%", pad=0.5,)

        
#     fig.text(-.05,0.280,'Neutral',fontsize=26,rotation=90)
#     fig.text(-.05,0.643,'Destabilizing',fontsize=26,rotation=90)
#     for ax in hm_axes:
#         ax.plot(
#             [-0.5,19.5],[19.5,19.5],
#             color='k',
#             linestyle='--',
#             linewidth=0.5
#         )
    
    
    fig.tight_layout()
    if isinstance(filename,str):
        plt.savefig(filename,bbox_inches='tight')
    plt.show()

def scatter_plots_ddG(df,value,coords,filename=None,lines=True,legend=True,ylabel=None,fontsize=8):
    fig = plt.figure(figsize=(2,2),dpi=300,constrained_layout=True)
    scatter_ax = fig.add_subplot()
    plot_scatter_ddG(
        df,value,coords,scatter_ax,
        filename=filename,lines=lines,legend=legend,
        ylabel=ylabel,
        fontsize=fontsize
    )
    if isinstance(filename,str):
        plt.savefig(filename,bbox_inches='tight')
    plt.show()
    
def plot_scatter_ddG(
    df, value, coords, scatter_ax,
    filename=None, lines=True, legend=True,
    ylabel=None,
    markersize=markersize,
    fontsize=8
):
    pearson_r,pearson_pval = pearsonr(
        df[~np.isnan(df['ddG'])][value],
        df[~np.isnan(df['ddG'])]['ddG'])
    spearman_r,spearman_pval = spearmanr(
        df[~np.isnan(df['ddG'])][value],
        df[~np.isnan(df['ddG'])]['ddG'])
    effect_to_c = {
        'Destabilizing':'#EC9C2C',
        'Neutral':'#374D89'
    }
    scatter_im = scatter_ax.scatter(
        df['ddG'],df[value],
        s=markersize,
        c=[effect_to_c[x] for x in df['effect']],
        zorder=2
    )
    if lines:
        scatter_ax.axvline(0,linestyle='--',color='k',zorder=1,linewidth=1)
        scatter_ax.axhline(0,linestyle='--',color='k',zorder=1,linewidth=1)
    scatter_ax.set_xlabel(
        r'stability effect, $\Delta\Delta G$',
        fontsize=fontsize,
#         rotation=90,
    )
    if ylabel == None:
        scatter_ax.set_ylabel(
            r'H-CNN prediction',
            fontsize=fontsize
        )
    else:
        scatter_ax.set_ylabel(
            ylabel,
            fontsize=fontsize,
        )
    if coords != None:
        scatter_ax.text(
            *coords,
            'Pearson corr. = {:.2f} \np-val = {:.2e}'.format(
                pearson_r,pearson_pval
            ) + '\n' + 'Spearman corr. = {:.2f} \np-val = {:.2e}'.format(
                spearman_r,spearman_pval
            )
        )


    if legend:
        scatter_ax.legend(
            handles=[
                mlines.Line2D(
                    [0], 
                    [0], 
                    marker='.', 
                    color='w', 
                    label='Scatter',
                    markerfacecolor='#EC9C2C', 
                    markersize=markersize
                ),
                mlines.Line2D(
                    [0], 
                    [0], 
                    marker='.', 
                    color='w', 
                    label='Scatter',
                    markerfacecolor='#374D89', 
                    markersize=markersize
                )
            ],
            labels=['destabilizing','neutral'],
            loc='center right',
            bbox_to_anchor=(2.1,.5)
        )



def dms_scatter_plot(
    df,
    dms_column, pred_column,
    dms_label=None, pred_label=None,
    filename=None, lines=False, legend=True,
    color='#84649B',
    markersize=4,
    fontsize=8
):
    fig = plt.figure(figsize=(2,2), dpi=300, constrained_layout=True)
    scatter_ax = fig.add_subplot()

    pearson_r, pearson_pval = pearsonr(
        df[np.logical_and(~np.isnan(df[dms_column]), ~np.isnan(df[pred_column]))][pred_column],
        df[np.logical_and(~np.isnan(df[dms_column]), ~np.isnan(df[pred_column]))][dms_column])
    spearman_r, spearman_pval = spearmanr(
        df[np.logical_and(~np.isnan(df[dms_column]), ~np.isnan(df[pred_column]))][pred_column],
        df[np.logical_and(~np.isnan(df[dms_column]), ~np.isnan(df[pred_column]))][dms_column])
    
    scatter_ax.scatter(
        df[dms_column], df[pred_column],
        s=markersize,
        c=color,
        zorder=2
    )

    if lines:
        scatter_ax.axvline(0, linestyle='--', color='k', zorder=1, linewidth=1)
        scatter_ax.axhline(0, linestyle='--', color='k', zorder=1, linewidth=1)
    
    dms_label = dms_label if dms_label is not None else dms_column
    scatter_ax.set_xlabel(
        dms_label,
        fontsize=fontsize
    )

    pred_label = pred_label if pred_label is not None else pred_column
    scatter_ax.set_ylabel(
        pred_label,
        fontsize=fontsize
    )

    if legend:
        scatter_ax.legend(
            handles=[
                mlines.Line2D(
                    [0], 
                    [0], 
                    marker='o', 
                    color='w', 
                    label='Scatter',
                    markerfacecolor=color, 
                    markersize=markersize
                )
            ],
            labels=['Pearson corr. = {:.2f} \np-val = {:.2e}'.format(pearson_r, pearson_pval)
                    + '\n' +
                    'Spearman corr. = {:.2f} \np-val = {:.2e}'.format(spearman_r, spearman_pval)],
            loc='center right',
            bbox_to_anchor=(2.1,.5)
        )
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

    return (pearson_r, pearson_pval), (spearman_r, spearman_pval)


def get_roc_curve(df,label_field,label,value_field):
    
    data_vals = df[value_field]
    pos_vals = df[df[label_field] == label][value_field]
    neg_vals = df[df[label_field] != label][value_field]
    
    n_pos = len(pos_vals)
    n_neg =len(neg_vals)
    n = n_pos + n_neg
    
    tp = np.zeros(shape=len(df)+2)
    fp = np.zeros(shape=len(df)+2)
    
    for i,thresh in enumerate(np.sort(data_vals)):
        tp[i+1] = np.count_nonzero(pos_vals <= thresh)/n_pos
        fp[i+1] = np.count_nonzero(neg_vals <= thresh)/n_neg
    tp[-1],fp[-1] = 1.,1.

    return fp,tp

def dms_roc_plot(df, dms_column, pred_column, dms_pos_value=None, dms_label=None, filename=None):

    dms_pos_value = dms_pos_value if dms_pos_value is not None else 1
    roc_wt_struct = get_roc_curve(
        df, dms_column, dms_pos_value, pred_column
    )
    auc_wt_struct = auc(*roc_wt_struct,)

    fig = plt.figure(figsize=(2,2),dpi=300,constrained_layout=True)

    roc_ax = fig.add_subplot()

    roc_ax.set_xlim(-0.05,1.05)
    roc_ax.set_ylim(-0.05,1.05)
    roc_ax.set_xlabel('FPR',fontsize=8)
    roc_ax.set_ylabel('TPR',fontsize=8)
    dms_label = dms_label if dms_label is not None else dms_column
    roc_ax.set_title(dms_label)

    roc_ax.plot([0,1],[0,1],linestyle=(0,(2,2,2)),c='k',linewidth=1.)

    roc_ax.plot(
        *roc_wt_struct,
        label="AUC = {:.2f}".format(auc_wt_struct),
        linewidth=2,
        c='#84649B',
        zorder=4
    #     alpha=0.7
    )

    roc_ax.set_xlim(-0.05,1.05)
    roc_ax.set_ylim(-0.05,1.05)
    roc_ax.legend(loc='center right') #,bbox_to_anchor=(2.5,.5))

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

    return auc_wt_struct
