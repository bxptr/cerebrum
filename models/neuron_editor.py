import torch

class NeuronEditor:
    """
    The NeuronEditor applies dynamic modifications to neuronal parameters.
    This can include disease-specific changes, scaling of conductances, or adding/removing channels.
    """

    def __init__(self, disease_name=None, disease_models=None):
        self.disease_name = disease_name
        self.disease_models = disease_models if disease_models is not None else {}

    def apply_disease_modifications(self, hh_params):
        """
        Apply disease-specific scaling to Hodgkin-Huxley parameters.
        hh_params: dictionary with 'gNa', 'gK', 'gL', 'gCa', etc.
        """
        if self.disease_name is None:
            # No disease modifications
            return hh_params

        if self.disease_name not in self.disease_models:
            print(f"Warning: Disease model '{self.disease_name}' not found. No changes applied.")
            return hh_params

        disease_config = self.disease_models[self.disease_name]
        # Scale conductances
        for param in ["gNa", "gK", "gL", "gCa"]:
            if param + "_scale" in disease_config:
                scale_factor = disease_config[param + "_scale"]
                hh_params[param] *= scale_factor
                hh_params[param] = float(hh_params[param])  # Ensure float type
                # Clamp to avoid extreme values
                hh_params[param] = max(0.0, hh_params[param])

        print(f"Disease model '{self.disease_name}' applied. Modified HH parameters: {hh_params}")
        return hh_params

    def apply_custom_edits(self, hh_params, custom_scales):
        """
        Apply arbitrary custom scaling to parameters.
        custom_scales: dict e.g. {"gNa":1.1, "gCa":0.9}
        """
        for param, scale in custom_scales.items():
            if param in hh_params:
                hh_params[param] *= scale
                hh_params[param] = float(hh_params[param])
                hh_params[param] = max(0.0, hh_params[param])
        return hh_params

