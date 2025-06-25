# Methane

This repository contains analysis utilities for methane concentration datasets.

The histogram analysis now annotates basic statistics (mean, median, standard
deviation, minimum and maximum) on the generated plot.

For K-means clustering in `lat_lon` mode, the analysis no longer outputs the
`tula_kmeans_lat_lon_clusters.csv` or `tula_kmeans_lat_lon_conteos.txt` files.
Instead, cluster counts appear in the title of the spatial scatter plot.
