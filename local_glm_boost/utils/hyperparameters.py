from typing import Union, Dict, Any, List

# Create a class called Hyperparameter that is basically a dict that can be initialized with a dict, a list, or a value
# and has a method to return a value for a given key.


class HyperparameterDict(dict):
    def __init__(
        self,
        parameter_value: Union[Dict[Union[int, str], Any], List[Any], Any],
        feature_names: List[Union[int, str]],
    ):
        super().__init__()
        p = len(feature_names)
        if isinstance(parameter_value, Dict):
            for j, feature_name in enumerate(feature_names):
                self[j] = parameter_value[feature_name]
        elif isinstance(parameter_value, List):
            for j in range(p):
                self[j] = parameter_value[j]
        else:
            for j in range(p):
                self[j] = parameter_value


class FeatureSelectionDict(dict):
    def __init__(
        self,
        feature_selection: Union[
            None,
            List[List[Union[int, str]]],
            Dict[Union[int, str], List[Union[int, str]]],
        ],
        feature_names: List[Union[int, str]],
    ):
        super().__init__()
        p = len(feature_names)
        if feature_selection is None:
            for j in range(p):
                self[j] = [k for k in range(p)]
        elif isinstance(feature_selection, List):
            for j in range(p):
                if isinstance(feature_selection[j][0], int):
                    self[j] = feature_selection[j]
                elif isinstance(feature_selection[j][0], str):
                    self[j] = [
                        feature_names.index(feature) for feature in feature_selection[j]
                    ]
        else:
            for j, feature_name in enumerate(feature_names):
                self[j] = [
                    feature_names.index(feature)
                    for feature in feature_selection[feature_name]
                ]
