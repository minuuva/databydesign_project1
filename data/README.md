# Project Data

All data files are stored in UVA OneDrive.

**Access the data here:** [OneDrive Data Folder](https://myuva-my.sharepoint.com/:f:/g/personal/qce2dp_virginia_edu/IgBsbp8bikwgRJCWBQ1DHrxQAdNoYJJKxI8g9_wMz_vkTgw?e=olv76k)

## Data Files

The following files are available in the OneDrive folder:

| File | Rows | Size | Description |
|------|------|------|-------------|
| movies.csv | 62,423 | 2.90 MB | Movie metadata (title, genres) |
| ratings.csv | 25,000,095 | 646.84 MB | User ratings (0.5-5.0 scale) |
| users.csv | 162,541 | 10.26 MB | User statistics (derived from ratings) |
| tags.csv | 1,093,360 | 37.01 MB | User-generated movie tags |
| links.csv | 62,423 | 1.31 MB | External IDs (IMDB, TMDB) |
| genome-scores.csv | 15,584,448 | 415.00 MB | Tag relevance scores |
| genome-tags.csv | 1,128 | 0.02 MB | Tag definitions |
| **TOTAL** | **42M+** | **1.087 GB** | **7 tables** |

## Parquet Versions

Compressed parquet versions are also available (198 MB total, 81.8% smaller).

## Data Source

MovieLens 25M Dataset from GroupLens Research, University of Minnesota.
- Official URL: https://grouplens.org/datasets/movielens/25m/
- Downloaded: March 10th 2026