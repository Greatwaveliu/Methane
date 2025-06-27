# Methane

This repository contains analysis utilities for methane concentration datasets.

The histogram analysis now annotates basic statistics (mean, median, standard
deviation, minimum and maximum) on the generated plot.

For K-means clustering in `lat_lon` mode, the analysis no longer outputs the
`tula_kmeans_lat_lon_clusters.csv` or `tula_kmeans_lat_lon_conteos.txt` files.
Cluster counts are shown in the title of the spatial scatter plot.

Similarly, for the `time_lat_lon` mode the analysis does not generate
`tula_kmeans_time_lat_lon_clusters.csv` or
`tula_kmeans_time_lat_lon_conteos.txt`. The cluster count statistics that would
normally be stored in the text file are displayed directly in the title of the
`tula_kmeans_time_lat_lon_space_scatter_basemap.png` plot.

The WCSS, Silhouette Score and Davies-Bouldin Index plots for all K-means modes
highlight the chosen cluster count using a dashed vertical line and a purple
scatter marker. Each plot now labels the marker with text like ``K = 3`` placed
to the right of the point, making the optimal cluster number easy to identify.

Each analysis also saves a final histogram. When methane data is available, the
plot now stacks the ``methane3`` distributions for all clusters in a single
histogram, making the concentration ranges easy to compare. Otherwise a simple
bar chart of cluster counts is produced.

For the ``lat_lon_methane`` mode four plots are produced:
``tula_kmeans_lat_lon_methane_metricas_vs_K.png``,
``tula_kmeans_lat_lon_methane_space_scatter_basemap.png``,
``tula_methane3_box_swarmplot_lat_lon_methane_cluster.png`` and
``tula_kmeans_lat_lon_methane_cluster_histogram.png``. No CSV or text files are
saved and intermediate methane histogram or boxplot images are not generated.

In ``time_lat_lon_methane`` mode five plots are produced:
``tula_kmeans_time_lat_lon_methane_metricas_vs_K.png``,
``tula_kmeans_time_lat_lon_methane_space_scatter_basemap.png``,
``tula_kmeans_time_lat_lon_methane_time_scatter.png``,
``tula_methane3_box_swarmplot_time_lat_lon_methane_cluster.png`` and
``tula_kmeans_time_lat_lon_methane_cluster_histogram.png``. Other plots as well
as CSV or text files are omitted.
