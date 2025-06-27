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
now highlight the chosen cluster count with a dashed vertical line, a scatter
marker and an annotation showing the value of **K**. This makes the optimal
cluster number easy to identify.
