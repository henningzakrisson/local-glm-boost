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
