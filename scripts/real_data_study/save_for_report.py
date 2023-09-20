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
    cont_features = ['VehPower', 'VehAge', 'Density', 'DrivAge', 'BonusMalus', 'Area']
    cat_features = ['VehGas', 'VehBrand', 'Region']

    kappa_table = "\centering \n"
    kappa_table += "\\begin{tabular}{c|lllllllll} \n"
    kappa_table += "\\toprule \n"

    feature_header = ["& \\rotatebox{45}{\\texttt{" + feature + "}}" for feature in cont_features + cat_features]
    kappa_table += ' '.join(feature_header) + " \\\\ \n"

    kappa_row = '\\kappa_j & '
    cont_features_kappa = ' & '.join([str(parameters.loc[feature, 'n_estimators']) for feature in cont_features])
    kappa_row += cont_features_kappa
    for cat_feature in cat_features:
        dummy_features = [feature for feature in features if feature.startswith(cat_feature)]
        min_kappa = parameters.loc[dummy_features, 'n_estimators'].min()
        max_kappa = parameters.loc[dummy_features, 'n_estimators'].max()
        kappa_range = f'({min_kappa}-{max_kappa})'
        kappa_row += ' & ' + kappa_range
    kappa_row += '\\\\ \n'
    kappa_table += kappa_row

    beta_row = '\\beta_{j0} & '
    cont_features_beta = ' & '.join([str(np.round(parameters.loc[feature, 'beta0'], 2)) for feature in cont_features])
    beta_row += cont_features_beta
    for cat_feature in cat_features:
        dummy_features = [feature for feature in features if feature.startswith(cat_feature)]
        min_beta = np.round(parameters.loc[dummy_features, 'beta0'].min(), 2)
        max_beta = np.round(parameters.loc[dummy_features, 'beta0'].max(), 2)
        beta_range = f'({min_beta}-{max_beta})'
        beta_row += ' & ' + beta_range
    beta_row += '\\\\ \n'
    kappa_table += beta_row
    kappa_table += '\\bottomrule \n'
    kappa_table += '\\end{tabular}'

    with open(tex_path + 'real_data_kappa.tex', 'w') as file:
        file.write(kappa_table)

    train_data = pd.read_csv(folder_path + 'train_data.csv', index_col=0)
    test_data = pd.read_csv(folder_path + 'test_data.csv', index_col=0)

    # Calculate deviance
    def deviance(y, w, z):
        log_y = np.zeros(len(y))
        log_y[y > 0] = np.log(y[y > 0])
        dev = w * np.exp(z) + y * (log_y - np.log(w) - z - 1)
        return 2 * dev.mean()

    dev_table = pd.DataFrame(columns=['Train', 'Test'], index=['Intercept', 'GLM', 'LocalGLMBoost', 'LocalGLMNet'],
                             dtype=float)

    dev_table.loc['Intercept', 'Train'] = deviance(train_data['y'], train_data['w'], train_data['z_0'])
    dev_table.loc['Intercept', 'Test'] = deviance(test_data['y'], test_data['w'], test_data['z_0'])
    dev_table.loc['GLM', 'Train'] = deviance(train_data['y'], train_data['w'], train_data['z_glm'])
    dev_table.loc['GLM', 'Test'] = deviance(test_data['y'], test_data['w'], test_data['z_glm'])
    dev_table.loc['LocalGLMBoost', 'Train'] = deviance(train_data['y'], train_data['w'],
                                                       train_data['z_local_glm_boost'])
    dev_table.loc['LocalGLMBoost', 'Test'] = deviance(test_data['y'], test_data['w'], test_data['z_local_glm_boost'])
    dev_table.loc['LocalGLMNet', 'Train'] = 23.728 / 100
    dev_table.loc['LocalGLMNet', 'Test'] = 23.945 / 100

    (dev_table * 100).round(3).to_latex(tex_path + 'real_data_deviance.tex')

    y_hat = test_data[['y']].copy()
    y_hat['Intercept'] = np.exp(test_data['z_0']) * test_data['w']
    y_hat['GLM'] = np.exp(test_data['z_glm']) * test_data['w']
    y_hat['LocalGLMBoost'] = np.exp(test_data['z_local_glm_boost']) * test_data['w']

    window_size = 1000
    y_hat_smooth = y_hat.sort_values('LocalGLMBoost').reset_index(drop=True).rolling(window_size).mean().dropna()

    y_hat_sample = y_hat_smooth.sample(1000)
    y_hat_sample.sort_values('LocalGLMBoost', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    y_hat_sample.plot(ax=ax, linewidth=2)

    plt.xlabel('Observation (ordered)');
    # Change the y-axis label to "Claim count (rolling average)"
    plt.ylabel('Claim count');
    # Save as tex
    tikzplotlib.save(tex_path + 'real_data_predictions.tex');

    feature_importances_og = pd.read_csv(folder_path + 'feature_importances.csv', index_col=0)
    feature_importances = feature_importances_og.copy().fillna(0)

    # Sum importance for categorical features
    for cat_feature in cat_features:
        dummy_features = [feature for feature in feature_importances.index if feature.startswith(cat_feature)]
        feature_importances.loc[cat_feature] = feature_importances.loc[dummy_features].sum(axis=0)
        feature_importances.drop(dummy_features, axis=0, inplace=True)

        feature_importances[cat_feature] = feature_importances[dummy_features].sum(axis=1)
        feature_importances.drop(dummy_features, axis=1, inplace=True)

    # Normalize
    feature_importances = feature_importances.div(feature_importances.sum(axis=0), axis=1).fillna(0)

    # Create a latex heatmap figure
    feature_heatmap = """\\begin{tikzpicture}
    \\begin{axis}[
        width = 0.8*\\textwidth,
        height = 0.6*\\textwidth,
        colormap/jet,
        colorbar,
        xtick=data,"""

    yticklabels = ", ".join([f"\\texttt{{{col}}}" for col in feature_importances.columns])
    yticklabels_line = f"    yticklabels={{{yticklabels}}},"
    feature_heatmap += yticklabels_line

    xticklabels = ", ".join([f"\\texttt{{{row}}}" for row in feature_importances.index])
    xticklabels_line = f"    xticklabels={{{xticklabels}}},"
    feature_heatmap += xticklabels_line

    feature_heatmap += """ytick=data,
        xticklabel style={rotate=45, anchor=north east, inner sep=0.5mm},
        nodes near coords={\pgfmathprintnumber[fixed, precision=2]\pgfplotspointmeta},
        nodes near coords style={text = gray, anchor=center, font=\\footnotesize},
        point meta=explicit,
        y dir=reverse
    ]
    \addplot [
    matrix plot*,"""

    mesh_col_line = f"    mesh/cols={len(feature_importances.columns)},"
    feature_heatmap += mesh_col_line
    mesh_row_line = f"    mesh/rows={len(feature_importances.index)},"
    feature_heatmap += mesh_row_line

    feature_heatmap += """]
    table [meta=value] {
    X Y value"""

    for j, feature in enumerate(feature_importances.columns):
        for i, value in enumerate(feature_importances[feature]):
            feature_heatmap += f"\n{j} {i} {np.round(value, 4)}"
        feature_heatmap += "\n"

    feature_heatmap += """
    };

    \end{axis}
    \end{tikzpicture}"""

    with open(tex_path + 'real_data_feature_importance.tex', 'w') as file:
        file.write(feature_heatmap)

