import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
import pandas as pd
import os

def save_tables_and_figures(run_id: int,save_to_git: bool):
    if save_to_git:
        folder_path = f'../../data/output_saved/run_{run_id}/'
    else:
        folder_path = f'../../data/output/run_{run_id}/'
    tex_path = folder_path + 'figures_and_tables/'
    os.makedirs(tex_path, exist_ok=True)

    parameters = pd.read_csv(folder_path + 'parameters.csv', index_col=0)

    features = parameters.index

    kappa_table = "\centering \n"
    kappa_table += "\\begin{tabular}{c|lllllllll} \n"
    kappa_table += "\\toprule \n"

    feature_header = ["& $" + feature.replace("X", "x_") + "$" for feature in features]
    kappa_table += ' '.join(feature_header) + " \\\\ \n"

    kappa_row = '$\\kappa_j$ & '
    features_kappa = ' & '.join([str(parameters.loc[feature, 'n_estimators']) for feature in features])
    kappa_row += features_kappa
    kappa_row += '\\\\ \n'
    kappa_table += kappa_row

    beta_row = '$\\beta_{j0}$ & '
    features_beta = ' & '.join([str(np.round(parameters.loc[feature, 'beta0'], 2)) for feature in features])
    beta_row += features_beta
    beta_row += '\\\\ \n'
    kappa_table += beta_row
    kappa_table += '\\bottomrule \n'
    kappa_table += '\\end{tabular}'

    with open(tex_path + 'simulated_data_kappa.tex', 'w') as file:
        file.write(kappa_table)

    train_data = pd.read_csv(folder_path + 'train_data.csv', index_col=0)
    test_data = pd.read_csv(folder_path + 'test_data.csv', index_col=0)

    # Calculate mse
    def mse(y, z):
        return np.mean((y - z) ** 2)

    mse_table = pd.DataFrame(columns=['Train', 'Test'], index=['True','Intercept', 'GLM', 'LocalGLMBoost', 'LocalGLMNet'],
                             dtype=float)

    mse_table.loc['True', 'Train'] = mse(train_data['y'], train_data['mu'])
    mse_table.loc['True', 'Test'] = mse(test_data['y'], test_data['mu'])
    mse_table.loc['Intercept', 'Train'] = mse(train_data['y'], train_data['z_0'])
    mse_table.loc['Intercept', 'Train'] = mse(train_data['y'], train_data['z_0'])
    mse_table.loc['Intercept', 'Test'] = mse(test_data['y'], test_data['z_0'])
    mse_table.loc['GLM', 'Train'] = mse(train_data['y'], train_data['z_glm'])
    mse_table.loc['GLM', 'Test'] = mse(test_data['y'], test_data['z_glm'])
    mse_table.loc['LocalGLMBoost', 'Train'] = mse(train_data['y'], train_data['z_local_glm_boost'])
    mse_table.loc['LocalGLMBoost', 'Test'] = mse(test_data['y'], test_data['z_local_glm_boost'])
    mse_table.loc['LocalGLMNet', 'Train'] = 1.0023
    mse_table.loc['LocalGLMNet', 'Test'] = 1.0047

    mse_table.round(4).to_latex(tex_path + 'simulated_data_mse.tex')

    feature_importances = pd.read_csv(folder_path + 'feature_importances.csv', index_col=0)

    feature_importances = feature_importances.div(feature_importances.sum(axis=1), axis=0).fillna(0)

    # Create a latex heatmap figure
    feature_heatmap = """\\begin{tikzpicture}
    \\begin{axis}[
        width = 0.8*\\textwidth,
        height = 0.6*\\textwidth,
        colormap/jet,
        colorbar,
        xtick=data,"""

    yticklabels = ", ".join([f"$\\beta_{{{col.replace('X', '')}}}$" for col in feature_importances.columns])
    yticklabels_line = f"    yticklabels={{{yticklabels}}},"
    feature_heatmap += yticklabels_line

    xticklabels = ", ".join([f"${{{row.replace('X', 'x_')}}}$" for row in feature_importances.index])
    xticklabels_line = f"    xticklabels={{{xticklabels}}},"
    feature_heatmap += xticklabels_line

    feature_heatmap += """ytick=data,
        nodes near coords={\pgfmathprintnumber[fixed, precision=2]\pgfplotspointmeta},
        nodes near coords style={text = gray, anchor=center, font=\\footnotesize},
        point meta=explicit,
        y dir=reverse
    ]
    \\addplot [
    matrix plot*, \n"""

    mesh_col_line = f"    mesh/cols={len(feature_importances.columns)}, \n"
    feature_heatmap += mesh_col_line
    mesh_row_line = f"    mesh/rows={len(feature_importances.index)}, \n"
    feature_heatmap += mesh_row_line

    feature_heatmap += """]
    table [meta=value] {
    X Y value"""

    for j, feature in enumerate(feature_importances.index):
        for i, value in enumerate(feature_importances.loc[feature]):
            feature_heatmap += f"\n{i} {j} {np.round(value, 4)}"
        feature_heatmap += "\n"

    feature_heatmap += """
    };

    \end{axis}
    \end{tikzpicture}"""

    with open(tex_path + 'simulated_data_feature_importance.tex', 'w') as file:
        file.write(feature_heatmap)

