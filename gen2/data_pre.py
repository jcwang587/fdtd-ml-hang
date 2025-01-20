import pandas as pd


em_data = pd.read_csv('data/em.csv')
qe_data = pd.read_csv('data/qe.csv')

fe_data = pd.read_csv('data/fe.csv')
factor_data = pd.read_csv('data/factor.csv')

# Rename the first column to "Distance" for clarity
fe_data.rename(columns={"Unnamed: 0": "Distance"}, inplace=True)

# Melt the dataframe to have Diameter, Distance, and Fe as columns
fe_long_format = fe_data.melt(
    id_vars=["Distance"],
    var_name="Diameter",
    value_name="Fe"
)

# Convert the Diameter and Distance columns to numeric types for consistency
fe_long_format["Diameter"] = pd.to_numeric(fe_long_format["Diameter"])
fe_long_format["Distance"] = pd.to_numeric(fe_long_format["Distance"])

# Assign appropriate names for each row representing abs, sca, and plasmon
factor_data["Parameter"] = ["abs", "sca", "plasmon"]

# Melt the factor data into a long format
factor_long_format = factor_data.melt(
    id_vars=["Parameter"],
    var_name="Diameter",
    value_name="Value"
)

# Convert the Diameter column to numeric
factor_long_format["Diameter"] = pd.to_numeric(factor_long_format["Diameter"])

# Pivot the factor_long_format to make it suitable for merging
factor_wide_format = factor_long_format.pivot_table(
    index="Diameter",
    columns="Parameter",
    values="Value"
).reset_index()

# Merge the new data with fe_long_format based on Diameter
fe_merged = fe_long_format.merge(factor_wide_format, on="Diameter", how="left")

# Change Distance to distance and Diameter to diameter
fe_merged.rename(columns={"Distance": "distance", "Diameter": "diameter", "Fe": "fe"}, inplace=True)

# Put fe the last column
fe_merged = fe_merged[["distance", "diameter", "abs", "sca", "plasmon", "fe"]]

# Output the merged data to a CSV file
fe_merged.to_csv('data/fe_merged.csv', index=False)



