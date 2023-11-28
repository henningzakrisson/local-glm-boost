import pandas as pd
import os
import json
import numpy as np
import yaml

run_id = 36
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
# Regression attentions
if prefix == "sim":
    features_for_attentions = features
elif prefix == "real":
    # Only consider the continuous features
    cat_features = ["VehGas", "VehBrand", "Region"]
    features_for_attentions = [
        feature for feature in features if not feature.startswith(tuple(cat_features))
    ]
relevant_attentions = [f"beta_{feature}" for feature in features_for_attentions]

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

# Calculate the variable importance and add it to this table
parameter_table.loc[
    "variableImportance", [attention[5:] for attention in relevant_attentions]
] = (
    train_data[relevant_attentions].abs().mean()
    / train_data[relevant_attentions].abs().mean().sum()
).values
parameter_table.loc[
    "variableImportance", [attention[5:] for attention in relevant_attentions]
] = parameter_table.loc[
    "variableImportance", [attention[5:] for attention in relevant_attentions]
].apply(
    lambda x: f"{x:.2f}"
)

if prefix == "sim":
    # Fix the column names
    parameter_table.columns = [
        f"$x_{i}$" for i in range(1, len(parameter_table.columns) + 1)
    ]
    # Normalize the variable importance

    # Make sure all numbers have zeros in the decimal places in the beta row
    parameter_table.loc["betaValue"] = parameter_table.loc["betaValue"].apply(
        lambda x: f"{x:.2f}"
    )

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
        parameter_table.loc["kappaValue", feature] = f"({kappa_min} -- {kappa_max})"
        beta_min = parameter_table.loc["betaValue", feature_dummies].min()
        beta_max = parameter_table.loc["betaValue", feature_dummies].max()
        parameter_table.loc["betaValue", feature] = f"({beta_min} -- {beta_max})"

        # The variable importance is not calculated for categorical variables
        parameter_table.loc["variableImportance", feature] = "-"

        # Drop the dummy features
        parameter_table = parameter_table.drop(feature_dummies, axis=1)

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


# Attention plots
test_data[features_for_attentions + relevant_attentions].to_csv(
    f"{data_path}/plot_data/{prefix}_attentions.csv", index=False
)
# GLM coefficient tex definitions for the attention plot
glm_parameters_string = ""
for j, feature in enumerate(features_for_attentions, start=1):
    this_glm_parameter = model_parameters["LocalGLMboost"]["beta0"][feature]
    if prefix == "sim":
        parameter_name = f"{prefix}Beta{int_to_roman(j)}"
    elif prefix == "real":
        parameter_name = f"{prefix}Beta{feature}"
    glm_parameters_string += f"\\def\\{parameter_name}{{{this_glm_parameter:.3f}}}\n"

with open(f"{data_path}/plot_data/{prefix}_glm_parameters.tex", "w") as f:
    f.write(glm_parameters_string)

# Prediction plots
if prefix == "sim":
    # Save mu, mu_GLM and mu_LocalGLMboost as a csv file
    test_data[["mu", "mu_GLM", "mu_LocalGLMboost"]].to_csv(
        f"{data_path}/plot_data/{prefix}_mu_hat.csv", index=False
    )

elif prefix == "real":
    window = 1000
    # Load the entire test data
    test_data = pd.read_csv(f"{data_path}/test_data.csv", index_col=0)
    # Sort the data by the predicted z of the LocalGLMboost model
    test_data = test_data.sort_values(by="z_LocalGLMboost").reset_index(drop=True)
    # Calculate rolling averages of the predicted exp(z) for models:
    # Intercept, GLM, GBM, LocalGLMboost
    rolling_average = (
        np.exp(test_data[["z_Intercept", "z_GLM", "z_GBM", "z_LocalGLMboost"]])
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )
    # Name the columns after the models
    rolling_average.columns = [col.replace("z_", "") for col in rolling_average.columns]
    # Also add rolling window of y divided by rolling window of w
    rolling_average["y"] = (
        test_data["y"].rolling(window=window, center=True, min_periods=1).mean()
        / test_data["w"].rolling(window=window, center=True, min_periods=1).mean()
    )
    # Sort so that y comes first
    rolling_average = rolling_average[["y", "Intercept", "GLM", "GBM", "LocalGLMboost"]]
    rolling_average.index.name = "index"
    # Sample 500 rows from the rolling average evenly
    rolling_average = rolling_average.iloc[:: len(rolling_average) // 500, :]
    # Save the rolling average as a csv file
    rolling_average.to_csv(f"{data_path}/plot_data/{prefix}_mu_hat.csv")


# Feature importance plot
feature_importance = pd.read_csv(f"{data_path}/feature_importance.csv", index_col=0)
if prefix == "real":
    # Sum the categorical features importance
    cat_features = ["VehGas", "VehBrand", "Region"]
    for feature in cat_features:
        dummy_features = [
            dummy_feature
            for dummy_feature in feature_importance.columns
            if dummy_feature.startswith(feature)
        ]
        feature_importance[feature] = feature_importance[dummy_features].sum(axis=1)
        feature_importance.loc[feature] = feature_importance.loc[dummy_features].sum()
        # Drop the dummy features
        feature_importance = feature_importance.drop(dummy_features, axis=1)
        feature_importance = feature_importance.drop(dummy_features, axis=0)

    # Make sure order is correct
    feature_importance = feature_importance[
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
# Normalize the feature importance
feature_importance = feature_importance.div(
    feature_importance.sum(axis=1), axis=0
).fillna(0)
feature_importance.index = range(len(feature_importance))
feature_importance.index.name = "beta"
feature_importance.columns = range(len(feature_importance.columns))
feature_importance = feature_importance.round(2)

# Melt for heatmap plot
fi_long = feature_importance.melt(ignore_index=False, var_name="feature").reset_index()
fi_long = fi_long.sort_values(by=["feature", "beta"])[["feature", "beta", "value"]]
fi_long.to_csv(f"{data_path}/plot_data/{prefix}_feature_importance.csv", index=False)
