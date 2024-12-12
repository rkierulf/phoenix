# Removes identical or too-close-together pseudo-time points from the SERGIO dataset

import pandas as pd

df = pd.read_csv("SERGIO_data.csv", header=None)

sorted_columns = df.iloc[-1].sort_values(ascending=True).index
df_sorted = df[sorted_columns]
pseudo_time = df.iloc[-1]
min_delta_t = 0.0001
time_val = -1.0

columns_to_keep = []
for i in range(len(pseudo_time)):
    if ((pseudo_time[i] - time_val) > min_delta_t):
        columns_to_keep.append(i)
        time_val = pseudo_time[i]

final_df = df_sorted[columns_to_keep]
final_df.to_csv("SERGIO_data2.csv", index=False)