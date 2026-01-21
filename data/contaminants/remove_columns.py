import pandas as pd

input_csv = "risultato_cepheidsWithPeriods_lt_1_filtrate.csv"
output_csv = "output.csv"

# List of columns you want to remove
cols_to_remove = ["p2error", "pferror", "p1oerror"]
df = pd.read_csv(input_csv)
df = df.drop(columns=cols_to_remove, errors="ignore")
df.to_csv(output_csv, index=False)