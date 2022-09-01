import numpy as np
import pandas as pd
from plotly.offline import init_notebook_mode

import plotly.graph_objs as go
from plotly.subplots import make_subplots

init_notebook_mode(connected=False)
import plotly.io as pio
pio.templates.default = 'plotly_dark'

def visualize_events(
    evtIDs,
    R,
    lpmt_x_array,
    lpmt_y_array,
    lpmt_z_array,
    lpmt_s_array,
    lpmt_fht_array,
    event_pos_x_array,
    event_pos_y_array,
    event_pos_z_array,
    event_edep_array,
    event_Redep_array,
    colorscale_pe=['rgb(150, 30, 70)', 'rgb(255, 200, 70)', 'rgb(255, 255, 100)'],
    colorscale_fht='ice',
    subplots=True,
    fht_bias=np.e**4.5,
    cc_show=False,
    cht_show=False,
    save=False
):
    
    coef = lambda boolean: 2 if boolean else 1
    
    theta = np.linspace(0,2*np.pi,100)
    phi = np.linspace(0,np.pi,100)
    x = np.outer(np.cos(theta),np.sin(phi))
    y = np.outer(np.sin(theta),np.sin(phi))
    z = np.outer(np.ones(100),np.cos(phi))

#     fig = go.Figure()
    if subplots:
        fig = make_subplots(
            rows=1, cols=2, column_widths=[0.5, 0.5],
            specs=[[{'type': 'scene'}, {'type': 'scene'}]]
        )
    else:
        fig = go.Figure()
        
    if cc_show:
        x_cc = np.zeros(len(evtIDs))
        y_cc = np.zeros(len(evtIDs))
        z_cc = np.zeros(len(evtIDs))

        for i in range(len(evtIDs)):
            x_cc[i] = np.sum(lpmt_x_array[i] * lpmt_s_array[i]) / np.sum(lpmt_s_array[i])
            y_cc[i] = np.sum(lpmt_y_array[i] * lpmt_s_array[i]) / np.sum(lpmt_s_array[i])
            z_cc[i] = np.sum(lpmt_z_array[i] * lpmt_s_array[i]) / np.sum(lpmt_s_array[i])
            
        R_cc = (x_cc**2 + y_cc**2 + z_cc**2)**0.5

    if cht_show:
        eps = 50
        x_cht = np.zeros(len(evtIDs))
        y_cht = np.zeros(len(evtIDs))
        z_cht = np.zeros(len(evtIDs))

        for i in range(len(evtIDs)):
            x_cht[i] = np.sum(lpmt_x_array[i] / (lpmt_fht_array[i] + eps)) / np.sum(1 / (lpmt_fht_array[i] + eps))
            y_cht[i] = np.sum(lpmt_y_array[i] / (lpmt_fht_array[i] + eps)) / np.sum(1 / (lpmt_fht_array[i] + eps))
            z_cht[i] = np.sum(lpmt_z_array[i] / (lpmt_fht_array[i] + eps)) / np.sum(1 / (lpmt_fht_array[i] + eps))
        
        R_cht = (x_cht**2 + y_cht**2 + z_cht**2)**0.5


    for i in range(len(evtIDs)):
        list1 = list(lpmt_s_array[i].round(2))
        list2 = list(lpmt_fht_array[i].round(2))
        str1 = ', '.join("nPE: " + str(list1[i]) + " FHT: " + str(list2[i]) for i in range(len(list1)))
        scatter_text = str1.split(", ")
        trace = lambda type_of_vis_obj, colorscale, name, colorbar_x, colorbar_title="", colorbar_tickprefix="1.e":\
                                 go.Scatter3d(
                                     x=lpmt_x_array[i], y=lpmt_y_array[i], z=lpmt_z_array[i],
                                     mode='markers',
                                     visible=(i == 0),
                                     text=scatter_text,# + " FHT: " + str(lpmt_fht_array[i])[:5],
                                     marker=dict(size=2.75,
                                                 color=type_of_vis_obj,
                                                 colorscale=colorscale,
                                                 colorbar=dict(x=colorbar_x,
                                                               title=colorbar_title,
                                                               tickprefix=colorbar_tickprefix),
                                                 opacity=1),
                                     name=name, showlegend=True)
        if subplots:
            fig.add_trace(trace(lpmt_s_array[i], colorscale_pe, "LPMT_nPE", 0.45, colorbar_title="Charge", colorbar_tickprefix=""), row=1, col=1)
            fig.add_trace(trace(np.log10(lpmt_fht_array[i]+fht_bias), colorscale_fht, "LPMT_FHT", 1.01, colorbar_title="FHT"), row=1, col=2)
        else:
            fig.add_trace(trace(lpmt_s_array[i], colorscale_pe, "LPMT_nPE"))

    for i in range(len(evtIDs)):
        trace = lambda legend: go.Scatter3d(
                                    x=[event_pos_x_array[i]],
                                    y=[event_pos_y_array[i]],
                                    z=[event_pos_z_array[i]],
                                    visible=(i == 0),
                                    mode='markers',
                                    text="Edep: " + str(event_edep_array[i])[:5] +\
                                         " Redep: " + str(event_Redep_array[i])[:5],
                                    marker=dict(size=7,
                                                color='white',
                                                # colorscale='portland',
                                                opacity=1),
                                    name='Edep = ' + str(event_edep_array[i])[:5] + " MeV",
                                    showlegend=legend)
        
        if subplots:
            fig.add_trace(trace(True), row=1, col=1)
            fig.add_trace(trace(False), row=1, col=2)
        else:
            fig.add_trace(trace(True))
            
    if cc_show:
        for i in range(len(evtIDs)):
            trace = lambda legend: go.Scatter3d(
                                    x=[x_cc[i]],
                                    y=[y_cc[i]],
                                    z=[z_cc[i]],
                                    visible=(i == 0),
                                    mode='markers',
                                    text="Edep: " + str(event_edep_array[i])[:5] +\
                                         " R_cc: " + str(R_cc[i])[:5],
                                    marker=dict(size=7,
                                                color='darkred',
                                                # colorscale='portland',
                                                opacity=1),
                                    name='Edep = ' + str(event_edep_array[i])[:5] + " MeV",
                                    showlegend=legend)

            if subplots:
                fig.add_trace(trace(False), row=1, col=1)
                fig.add_trace(trace(False), row=1, col=2)
            else:
                fig.add_trace(trace(True))
    
    if cht_show:
        for i in range(len(evtIDs)):
            trace = lambda legend: go.Scatter3d(
                                x=[x_cht[i]],
                                y=[y_cht[i]],
                                z=[z_cht[i]],
                                visible=(i == 0),
                                mode='markers',
                                text="Edep: " + str(event_edep_array[i])[:5] +\
                                     " R_cht: " + str(R_cht[i])[:5],
                                marker=dict(size=7,
                                            color='royalblue',
                                            # colorscale='portland',
                                            opacity=1),
                                name='Edep = ' + str(event_edep_array[i])[:5] + " MeV",
                                showlegend=legend)

            if subplots:
                fig.add_trace(trace(False), row=1, col=1)
                fig.add_trace(trace(False), row=1, col=2)
            else:
                fig.add_trace(trace(True))

    for i in range(len(evtIDs)):
        trace = go.Surface(
                    x=x * R * 0.99,
                    y=y * R* 0.99,
                    z=z * R* 0.99,
                    opacity=0.3,
                    visible=(i == 0),
                    showscale=False,
                    colorscale=['rgb(2, 2, 2)', 'rgb(4, 4, 4)'],
                    name='')
        
        if subplots:
            fig.add_trace(trace, row=1, col=1)
            fig.add_trace(trace, row=1, col=2)
        else:
            fig.add_trace(trace)

    buttons = []
    k = coef(subplots)
    for N in range(0, len(evtIDs)): 
        buttons.append(
          dict(
              args=['visible', [False]*k*N + k*[True] + [False]*k*(len(evtIDs) - 1 - N)],
              label='EvtID = {}'.format(evtIDs[N]),
              method='restyle'
            )
        )            

    fig.update_layout(
      updatemenus=list([
        dict(
          x=0.05,
          y=1.1,
          yanchor='top',
          buttons=buttons
        ),
      ]),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1
        )
#       scene_camera_eye=dict(x=1, y=1, z=1)
    )

    fig.show()
    if save:
        pio.write_image(fig, 'plots/visualize_events.pdf', width=950, height=600)
