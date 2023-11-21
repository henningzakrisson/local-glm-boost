import pandas as pd
import os
import json
import numpy as np
import yaml

run_id = 20
random_state = 1


def int_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num


# Create a plot_data folder if it doesn't exist
data_path = f"../data/output/run_{run_id}"
os.makedirs(f"{data_path}/plot_data", exist_ok=True)

# Check data type
with open(f"{data_path}/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
if config["data_source"] == "simulation":
    prefix = "sim"
elif config["data_source"] == "real":
    prefix = "real"

# Load the test data
test_data = pd.read_csv(f"{data_path}/test_data.csv", index_col=0).sample(
    n=500, random_state=random_state
)
train_data = pd.read_csv(f"{data_path}/train_data.csv", index_col=0)

features = [
    col
    for col in test_data.columns
    if not col.startswith("z_")
    and not col.startswith("mu_")
    and not col.startswith("beta_")
    and col != "y"
    and col != "w"
    and col != "z"
    and col != "mu"
]
attentions = [f"beta_{feature}" for feature in features]

# Loss table
mse_table = pd.read_csv(f"{data_path}/loss_table.csv", index_col=0)
mse_table.index.name = "Model"
mse_table.columns = ["Train", "Test"]
if prefix == "sim":
    mse_table = mse_table.drop("GBM")
    mse_table.loc["LocalGLMnet"] = [1.0023, 1.0047]
elif prefix == "real":
    mse_table = 100 * mse_table.drop("True")
    mse_table.loc["LocalGLMnet"] = [23.728, 23.945]
mse_table = mse_table.round(4)
mse_table = mse_table.applymap(lambda x: f"{x:.3f}")
mse_table.to_csv(f"{data_path}/plot_data/{prefix}_loss.csv")

# Model parameters
with open(f"{data_path}/model_parameters.json") as f:
    model_parameters = json.load(f)
parameter_table = pd.DataFrame(
    index=pd.Index(["kappaValue", "betaValue"], name="parameter"), columns=features
)
parameter_table.loc["kappaValue"] = model_parameters["LocalGLMboost"]["n_estimators"]
parameter_table.loc["betaValue"] = np.array(
    list(model_parameters["LocalGLMboost"]["beta0"].values())
).round(3)

# Also calculate the variable importance and add it to this table
for feature in features:
    parameter_table.loc["variableImportance", feature] = (
        train_data[f"beta_{feature}"].abs().mean()
    )

if prefix == "sim":
    # Fix the column names
    parameter_table.columns = [
        f"$x_{i}$" for i in range(1, len(parameter_table.columns) + 1)
    ]
    # Normalize the variable importance
    parameter_table.loc["variableImportance"] = parameter_table.loc[
        "variableImportance"
    ].div(parameter_table.loc["variableImportance"].sum())
    # Make sure all numbers have zeros in the decimal places in the beta row
    parameter_table.loc["betaValue"] = parameter_table.loc["betaValue"].apply(
        lambda x: f"{x:.3f}"
    )
    parameter_table.loc["variableImportance"] = parameter_table.loc[
        "variableImportance"
    ].apply(lambda x: f"{x:.2f}")
elif prefix == "real":
    # If the data is real, we need to make ranges for categorical variables
    # Make sure all numbers have zeros in the decimal places in the beta row
    parameter_table.loc["betaValue"] = parameter_table.loc["betaValue"].apply(
        lambda x: f"{x:.2f}"
    )
    cat_features = ["VehGas", "VehBrand", "Region"]
    for feature in cat_features:
        feature_dummies = [
            dummy_feature
            for dummy_feature in features
            if dummy_feature.startswith(feature)
        ]
        kappa_min = parameter_table.loc["kappaValue", feature_dummies].min()
        kappa_max = parameter_table.loc["kappaValue", feature_dummies].max()
        parameter_table.loc["kappaValue", feature] = f"({kappa_min}-{kappa_max})"
        beta_min = parameter_table.loc["betaValue", feature_dummies].min()
        beta_max = parameter_table.loc["betaValue", feature_dummies].max()
        parameter_table.loc["betaValue", feature] = f"({beta_min}-{beta_max})"

        # The variable importance is the sum of the variable importance of the dummy features
        parameter_table.loc["variableImportance", feature] = parameter_table.loc[
            "variableImportance", feature_dummies
        ].sum()

        # Drop the dummy features
        parameter_table = parameter_table.drop(feature_dummies, axis=1)
    # Normalize the variable importance
    parameter_table.loc["variableImportance"] = parameter_table.loc[
        "variableImportance"
    ].div(parameter_table.loc["variableImportance"].sum())
    parameter_table.loc["variableImportance"] = parameter_table.loc[
        "variableImportance"
    ].apply(lambda x: f"{x:.2f}")

    # Make sure the order is correct
    parameter_table = parameter_table[
        [
            "VehPower",
            "VehAge",
            "Density",
            "DrivAge",
            "BonusMalus",
            "Area",
            "VehGas",
            "VehBrand",
            "Region",
        ]
    ]

#

# transpose the table
parameter_table = parameter_table.transpose()
# Name the index "feature"
parameter_table.index.name = "featureName"
parameter_table.to_csv(f"{data_path}/plot_data/{prefix}_parameters.csv")

# Regression attentions
test_data[features + attentions].to_csv(
    f"{data_path}/plot_data/{prefix}_attentions.csv", index=False
)
# GLM coefficient tex definitions for the attention plot
glm_parameters_string = ""
for j, feature in enumerate(features, start=1):
    this_glm_parameter = model_parameters["LocalGLMboost"]["beta0"][feature]
    glm_parameters_string += f"\\def\\beta{int_to_roman(j)}{{{this_glm_parameter}}}\n"

with open(f"{data_path}/plot_data/{prefix}_glm_parameters.tex", "w") as f:
    f.write(glm_parameters_string)

# Prediction plots
if prefix == "sim":
    # Save mu, mu_GLM and mu_LocalGLMboost as a csv file
    test_data[["mu", "mu_GLM", "mu_LocalGLMboost"]].to_csv(
        f"{data_path}/plot_data/{prefix}_mu_hat.csv", index=False
    )

# Feature importance plot
feature_importance = pd.read_csv(f"{data_path}/feature_importance.csv", index_col=0)
feature_importance = feature_importance.div(
    feature_importance.sum(axis=1), axis=0
).fillna(0)
feature_importance.index = range(len(features))
feature_importance.index.name = "beta"
feature_importance.columns = range(len(features))
feature_importance = feature_importance.round(2)
fi_long = feature_importance.melt(ignore_index=False, var_name="feature").reset_index()
fi_long = fi_long.sort_values(by=["feature", "beta"])[["feature", "beta", "value"]]
fi_long.to_csv(f"{data_path}/plot_data/{prefix}_feature_importance.csv", index=False)
