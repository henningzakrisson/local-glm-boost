import pandas as pd
import os
import json

simulated_run_id = 14
sim_data_path = f"../data/output/run_{simulated_run_id}"

# Create a plot_data folder if it doesn't exist
os.makedirs(f"{sim_data_path}/plot_data", exist_ok=True)

# For the mean comparison, load the test data
test_data = pd.read_csv(f"{sim_data_path}/test_data.csv")
# Randomly sample 500 rows from the test data
test_data = test_data.sample(n=500, random_state=1)
# Save mu, mu_GLM and mu_LocalGLMboost as a csv file
test_data[["mu", "mu_GLM", "mu_LocalGLMboost"]].to_csv(
    f"{sim_data_path}/plot_data/sim_mean_hat.csv", index=False
)

# For the regression attentions, save "Xj" and "beta_Xj"
# for j = 1,...,8 as a csv file
features = [f"X{j}" for j in range(1, 9)]
attentions = [f"beta_X{j}" for j in range(1, 9)]
test_data[features + attentions].to_csv(
    f"{sim_data_path}/plot_data/sim_attentions.csv", index=False
)

# Load the GLM coefficients
with open(f"{sim_data_path}/model_parameters.json") as f:
    model_parameters = json.load(f)
words = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
}
glm_parameters_string = ""
for j in range(1, 9):
    this_glm_parameter = model_parameters["LocalGLMboost"]["beta0"][f"X{j}"]
    glm_parameters_string += f"\\def\\beta{words[j]}{{{this_glm_parameter}}}\n"
# Save the string as a tex file
with open(f"{sim_data_path}/plot_data/sim_glm_parameters.tex", "w") as f:
    f.write(glm_parameters_string)

# Save the MSE table as a tex file table
loss_table = pd.read_csv(f"{sim_data_path}/loss_table.csv", index_col=0)
table_string = ""
for model in ["True", "Intercept", "GLM", "LocalGLMboost"]:
    train_loss = loss_table.loc[model, "train"]
    test_loss = loss_table.loc[model, "test"]
    table_string += f"{model} & {train_loss:.4f} & {test_loss:.4f} \\\\ \n"
# LocalGLMnet
train_loss = 1.0023
test_loss = 1.0047
table_string += f"LocalGLMnet & {train_loss:.4f} & {test_loss:.4f} \\\\"
# Save the string as a tex file
with open(f"{sim_data_path}/plot_data/sim_loss.tex", "w") as f:
    f.write(table_string)

# Save a table of n_estimators and beta0 for the LocalGLMboost model
# as a tex file table
# First a row with the n_estimators
kappa_table = "$\\kappa_j$ & "
for j in range(1, 9):
    n_estimators = model_parameters["LocalGLMboost"]["n_estimators"][f"X{j}"]
    kappa_table += f"{n_estimators} & "
kappa_table = kappa_table[:-2] + "\\\\ \n"
# Then a row with the beta0
kappa_table += "$\\widehat{\\beta}_{j}^{0}$ & "
for j in range(1, 9):
    beta0 = model_parameters["LocalGLMboost"]["beta0"][f"X{j}"]
    # If the beta0 when rounded to 3 decimals is 0, then write 0
    if round(beta0, 3) == 0.0:
        beta0 = 0
    kappa_table += f"{beta0:.3f} & "
kappa_table = kappa_table[:-2] + "\\\\ \n"
# Save the string as a tex file
with open(f"{sim_data_path}/plot_data/sim_parameters.tex", "w") as f:
    f.write(kappa_table)
