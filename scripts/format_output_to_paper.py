import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Load data (3 is the current simulation run)
run_id = 3
folder_path = f"../data/results/run_{run_id}/"

feature_importance = pd.read_csv(folder_path + "feature_importance.csv", index_col = 0)
feature_importance_light = pd.read_csv(folder_path + "feature_importance_light.csv", index_col = 0)
n_estimators = pd.read_csv(folder_path + "n_estimators.csv", index_col = 0)
beta0 = pd.read_csv(folder_path + "beta0.csv", index_col = 0)
mse = pd.read_csv(folder_path + "MSE.csv", index_col = 0)
tuning_loss = {
    'train': pd.read_csv(folder_path + "tuning_loss_train.csv", index_col = 0),
    'valid': pd.read_csv(folder_path + "tuning_loss_valid.csv", index_col = 0)
}
mu_hat = pd.read_csv(folder_path + "mu_hat.csv", index_col = 0)
beta_hat = {
    model_name: pd.read_csv(f"{folder_path}/beta_hat/{model_name}.csv", index_col=0) for model_name in ["true", "local_glm_boost", "local_glm_boost_light"]
}

p = len(beta0)

def format_model_name(name):
    """Format the model name to have initial capital letters.
    Underscores are removed and the following letter capitalized. 'glm' is made upper case."""

    name = ''.join(word.title() for word in name.split('_'))
    if 'glm' in name.lower():
        start = name.lower().find('glm')
        name = name[:start] + 'GLM' + name[start+3:]

    return name

def df_to_latex(df):
    """Convert the DataFrame to a LaTeX tabular environment."""

    df.index = df.index.map(format_model_name)
    latex_table = df.to_latex(column_format='l|cc', float_format="%.4f")

    # Insert \hline after the header line
    latex_table = latex_table.replace('\n', '\n\\hline', 1)
    # Add \hline at the end of the table
    latex_table = latex_table.rstrip() + '\n\\hline\n'

    return latex_table

mse_tex = df_to_latex(mse)

with open('simulated_data_mse.tex', 'w') as f:
    f.write(mse_tex)

def format_feature_name(name):
    """Format the feature name to match LaTeX style."""
    return f'$x_{{{name+1}}}$'

def df_to_latex(n_estimators, beta0):
    """Convert the DataFrame and Series to a LaTeX tabular environment."""

    n_estimators.index = n_estimators.index.map(format_feature_name)
    beta0.index = beta0.index.map(format_feature_name)

    # Create a combined DataFrame
    df = pd.DataFrame({
        'Feature': n_estimators.index.values,
        r'$\kappa_j$': n_estimators[n_estimators.columns[0]].values.flatten(),
        r'$\beta_{j0}$': beta0['local_glm_boost'].values.flatten()
    }).set_index('Feature')

    latex_table = df.to_latex(column_format='c|cr', bold_rows=False, float_format="%.3f", escape=False, index_names=False)

    return latex_table

latex_table = df_to_latex(n_estimators, beta0)
with open('simulated_data_kappa.tex', 'w') as f:
    f.write(latex_table)

def df_to_heatmap(df):
    beta_indices = [i.split('_')[-1] for i in df.index]
    y_ticks = ', '.join(["$\\beta_" + str(int(i)+ 1) + '$' for i in beta_indices])
    x_indices = [int(i.split('_')[-1])+1 for i in df.columns]
    x_ticks = ', '.join(["$x_" + str(i) + '$' for i in x_indices])

    tikz_code = "\\begin{tikzpicture}\n"
    tikz_code += "\\begin{axis}[\n"
    tikz_code += "    colormap/jet,\n"
    tikz_code += "    colorbar,\n"
    tikz_code += "    xtick=data,\n"
    tikz_code += "yticklabels={" + y_ticks + "},\n"
    tikz_code += "xticklabels={" + x_ticks + "},\n"
    tikz_code += "    ytick=data,\n"
    tikz_code += "    nodes near coords={\\pgfmathprintnumber[fixed, precision=2]\\pgfplotspointmeta},\n"
    tikz_code += "    nodes near coords style={text = gray, anchor=center, font=\\footnotesize},\n"
    tikz_code += "    point meta=explicit,\n"
    tikz_code += "    y dir=reverse\n"
    tikz_code += "]\n"
    tikz_code += "\\addplot [\n"
    tikz_code += "matrix plot*,\n"
    tikz_code += f"    mesh/cols={len(df.columns)},\n"
    tikz_code += "]\n"
    tikz_code += "table [meta=value] {\n"
    tikz_code += "X Y value\n"
    for row in range(len(df)):
        for col in range(len(df.columns)):
            tikz_code += f"{col} {row} {df.iloc[row, col].round(2)}\n"
        tikz_code += '\n'
    tikz_code += "};\n"
    tikz_code += "\end{axis}\n"
    tikz_code += "\end{tikzpicture}\n"
    return tikz_code

# Generate LaTeX TikZ code
feature_imp_tex = df_to_heatmap(feature_importance)
with open('simulated_data_feature_importance.tex', 'w') as f:
    f.write(feature_imp_tex)