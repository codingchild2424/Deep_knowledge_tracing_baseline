from visualizers.roc_auc_visualizer import roc_curve_visualizer
from visualizers.personal_pred_visualizer import personal_pred_visualizer

def get_visualizers(y_true_record, y_score_record,model, model_path, test_loader, device, config):
    roc_curve_visualizer(y_true_record, y_score_record, config.model_name)
    personal_pred_visualizer(model, model_path, test_loader, device, config.model_name)