import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import numpy as np
import plotly

#ggjkgjkfkjgkj
#https://plot.ly/python/carpet-plot/
#https://plot.ly/python/reference/

def carpet_plot(plotdata, y_axis_index, main_contour_indices, constraint_indices, variables_names_with_units):

    z_contour = np.array(plotdata[constraint_indices[0]:constraint_indices[0] + 1])[0]
    aa = 0
    t1_arrays_a = plotdata[main_contour_indices[1]:main_contour_indices[1] + 1][:, 0, :][0]
    t1_arrays_b = plotdata[main_contour_indices[0]:main_contour_indices[0] + 1][:, :, 0][0]
    t1_arrays_z = np.array(plotdata[constraint_indices[0]:constraint_indices[0] + 1])[0]
    if len(t1_arrays_a) != len(t1_arrays_z[0]):
        print('length of a should match the inner dimension of the data arrays')
        return
    if len(t1_arrays_b) != len(t1_arrays_z):
        print('length of b should match the the outer dimension of the data arrays')
        return
    
    #starting contour level value
    t1_start = int(np.amin(z_contour))
    # the end contour level value. Must be more than contours.start
    t1_end = int(np.amax(z_contour))
    #step between each contour level
    t1_size = (np.amax(z_contour)-np.amin(z_contour))/6
    t1_size = int(round(t1_size))

    trace1 = go.Contourcarpet(

        b = t1_arrays_b,
        a = t1_arrays_a,
        z = t1_arrays_z,


        autocontour=False, #whether or not the contour level attributes are picked
        contours=dict(
            start = t1_start, 
            end = t1_end,  
            size = t1_size,
            coloring="lines", # dash options include "lines" | "none"
            showlabels=True,

            labelfont=dict(
                family='Arial, sans-serif',
                size=10,
                color='blue',
            )

        ),
        line=dict(
            width=1,
            color="red",
            smoothing=1,
            dash='dash'  # dash options include 'dash', 'dot', and 'dashdot'
        ),

        colorbar=dict(
            len = 0.4,
            y=0.25 #Sets the y position of the color bar (in plot fraction).
        )
    )

    t2_arrays_a = plotdata[main_contour_indices[1]:main_contour_indices[1] + 1][:, 0, :][0]
    t2_arrays_b = plotdata[main_contour_indices[0]:main_contour_indices[0] + 1][:, :, 0][0]

    trace2 = go.Carpet(

        b = t2_arrays_b,
        a = t2_arrays_a,

        x=[[1, 2, 3, 4, 5],
           [1, 1.9, 2.8, 3.9, 4.9],
           [1, 1.8, 2.7, 3.8, 4.8],
           [1, 1.7, 2.6, 3.7, 4.7],
           [1, 1.6, 2.5, 3.6, 4.6]],

        y=plotdata[y_axis_index:y_axis_index + 1][0],  # Range

        aaxis=dict(
            tickprefix='T/W = ',
            type='linear',
            smoothing=0,
            minorgridcount=9,
            # minorgridwidth = 0.6,
            # minorgridcolor = 'pink',
            gridcolor='magenta',  # This changes the color of the grid but not the edges
            color='magenta',
            gridwidth=2,
            title="<b>W/S<br>(psf)</b>",

            titlefont=dict(
                family='Arial, sans-serif',
                size=15,
                color='black'
            ),
            tickfont=dict(
                family='Arial, sans-serif',
                size=17,
                color='black'
            ),

        ),

        baxis=dict(
            tickprefix='W/S = ',
            ticksuffix = 'psf',
            type='linear',
            smoothing=0,
            minorgridcount=9,
            # minorgridwidth = 0.6,
            # minorgridcolor = 'pink',
            gridcolor='orange',  #changes the color of the grid but not the edges
            color='orange',
            #changes the color of the edges of the grid.
            gridwidth=2,
            tickfont=dict(
                family='Old Standard TT, serif',
                size=17,
                color='black'
            ),

            title="T/W",
            titlefont=dict(
                family='Arial, sans-serif',
                size=15,
                color='black'
            ),

        )
    )

    data = [trace1, trace2]
    layout = go.Layout(
        plot_bgcolor='white',
        paper_bgcolor='white',  # Plot is a subset of paper!
        title="Carpet Plot for sensitivity parameters",
        margin=dict(  # This is to dictate the extent of margins around the plot
            t=60,
            r=50,
            b=50,
            l=70
        ),
        yaxis=dict(
            tickprefix = 'Range = ',
            ticksuffix = 'nmi',
            ticks='outside',
            showgrid=True,
            showticklabels=True,

            # minorgridcount = 1,
            # minorgridwidth = 0.6,
            # minorgridcolor = 'pink',
            gridwidth=2,
            gridcolor='#bdbdbd',
            autorange=True,
            title="<b>Range <br>(nmi)</b>",
            tickangle=0,
            tickwidth=2,
            tickcolor='#000',
            tickfont=dict(
                family='Old Standard TT, serif',
                size=17,
                color='black'
            ),
            #  range = [4000,5000]     # for our own range

        ),

        xaxis=dict(
            ticks='outside',
            tick0=0,
            showgrid=True,
            showticklabels=True,
            autorange=True
            #   rangemode='nonnegative'
            # 	range = [-5,6]	                   # for our own range

        )

    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='carpet_input_format.html')
    return;



WS = np.array([[10, 10, 10, 10, 10],
               [9.5, 9.5, 9.5, 9.5, 9.5],
               [9, 9, 9, 9, 9],
               [8.5, 8.5, 8.5, 8.5, 8.5],
               [8, 8, 8, 8, 8]])

TW = np.array([[.4, .425, .45, .475, .5],
               [.4, .425, .45, .475, .5],
               [.4, .425, .45, .475, .5],
               [.4, .425, .45, .475, .5],
               [.4, .425, .45, .475, .5]])

R = np.array([[6800, 6825, 6750, 6625, 6450],
              [6700, 6700, 6675, 6575, 6450],
              [6550, 6575, 6575, 6525, 6425],
              [6350, 6425, 6450, 6450, 6375],
              [6125, 6250, 6300, 6325, 6300]])

BFL = np.array([[14000, 12000, 10750, 9750, 9000], 
                [12000, 12050, 10100, 9250, 8750],
                [11200, 10750, 8750, 8900, 8300],
                [11000, 10000, 9100, 7500, 7800],
                [10500, 9800, 8800, 8000, 8200]])

FV = np.array([[6.5, 5.7, 5, 4, 3.2],
               [5, 4.5, 4, 3.1, 3],
               [3, 2.7, 2.3, 2, 1.9],
               [1.5, 1.4, 1.3, 1.2, 1.1],
               [.2, .15, .1, .05, 0]]) 

#input variables
plotdata = np.array([WS, TW, R, BFL, FV])
y_axis_index = 2
main_contour_indices = [0, 1]
constraint_indices = [3, 4]
variables_names_with_units = ['W/S (psf)', 'T/W (-)', 'Range (nmi)', 'BFL (ft)', 'Fuel Vol (m^3)']


carpet_plot(plotdata, y_axis_index, main_contour_indices, constraint_indices, variables_names_with_units)

