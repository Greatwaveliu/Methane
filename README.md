# Methane

This repository contains analysis utilities for methane concentration datasets.

The histogram analysis now annotates basic statistics (mean, median, standard
deviation, minimum and maximum) on the generated plot.

For K-means clustering in `lat_lon` mode, the analysis no longer outputs the
`tula_kmeans_lat_lon_clusters.csv` or `tula_kmeans_lat_lon_conteos.txt` files.
Cluster counts are shown in the title of the spatial scatter plot. Three plots
are produced:
``tula_kmeans_lat_lon_metricas_vs_K.png``,
``tula_kmeans_lat_lon_space_scatter_basemap.png`` and
``tula_methane3_stacked_hist_lat_lon_cluster.png``.

Similarly, for the `time_lat_lon` mode the analysis does not generate
`tula_kmeans_time_lat_lon_clusters.csv` or
`tula_kmeans_time_lat_lon_conteos.txt`. The cluster count statistics that would
normally be stored in the text file are displayed directly in the title of the
`tula_kmeans_time_lat_lon_space_scatter_basemap.png` plot. Four plots are
produced:
``tula_kmeans_time_lat_lon_metricas_vs_K.png``,
``tula_kmeans_time_lat_lon_space_scatter_basemap.png``,
``tula_kmeans_time_lat_lon_time_scatter.png`` and
``tula_methane3_stacked_hist_time_lat_lon_cluster.png``.

The WCSS, Silhouette Score and Davies-Bouldin Index plots for all K-means modes
highlight the chosen cluster count using a dashed vertical line and a purple
scatter marker. Each plot now labels the marker with text like ``K = 3`` placed
to the right of the point, making the optimal cluster number easy to identify.

For the ``lat_lon_methane`` mode four plots are produced:
``tula_kmeans_lat_lon_methane_metricas_vs_K.png``,
``tula_kmeans_lat_lon_methane_space_scatter_basemap.png``,
``tula_methane3_box_swarmplot_lat_lon_methane_cluster.png`` and
``tula_methane3_stacked_hist_lat_lon_methane_cluster.png``. No CSV or text
files are saved and intermediate methane boxplot images are not generated.

In ``time_lat_lon_methane`` mode five plots are produced:
``tula_kmeans_time_lat_lon_methane_metricas_vs_K.png``,
``tula_kmeans_time_lat_lon_methane_space_scatter_basemap.png``,
``tula_kmeans_time_lat_lon_methane_time_scatter.png``,
``tula_methane3_box_swarmplot_time_lat_lon_methane_cluster.png`` and
``tula_methane3_stacked_hist_time_lat_lon_methane_cluster.png``. Other plots
as well as CSV or text files are omitted.

The DFA (Detrended Fluctuation Analysis) plot labels the y-axis as
``Función de fluctuación F(n)`` and includes a short note summarizing common
interpretations of the Hurst exponent values.

The PSA (Power Spectral Analysis) now generates only
``loglog_psd_with_beta.png``. This plot contains a brief guide to common values
of the spectral exponent:
``β=0: white noise``, ``β=1: pink noise (1/f)``, ``β=2: Brownian noise``. The
plot also includes the calculated β value with a short explanation such as
``Pink noise-like process`` or ``Brownian or red noise`` depending on the
result.

The LSTM analysis returns a single figure ``LSTM.png`` that overlays the
training and testing predictions with the historical data and appends the
future forecast for six months. The forecasted period (application stage) is
highlighted in the plot. To keep the curves visible the legend uses a smaller
font size, and the figure is large enough to display clearly at 100% zoom.
