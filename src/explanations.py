import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

def get_shap_values(model, data):
    """Generate SHAP values for model interpretation."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.summary_plot(shap_values, data)

def get_lime_explanation(model, data, instance):
    """Generate a LIME explanation for a single prediction."""
    explainer = LimeTabularExplainer(data.values, mode="classification")
    explanation = explainer.explain_instance(instance, model.predict_proba)
    explanation.show_in_notebook(show_all=False)
