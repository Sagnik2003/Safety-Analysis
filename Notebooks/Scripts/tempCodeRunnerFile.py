# Example usage
    # y_true = np.array([0, 1, 1, 0, 1])
    # y_pred = np.array([0, 1, 0, 0, 1])
    # y_prob = np.array([0.1, 0.9, 0.4, 0.3, 0.8])

    # evaluator = evaluation_methods(y_true, y_pred, y_prob)
    # metrics_df = evaluator.compute_metrics()
    # print(metrics_df)

    # cm_file = evaluator.save_confusion_matrix()
    # print(f"Confusion matrix saved to {cm_file}")

    # roc_file = evaluator.plot_roc_curve()
    # print(f"ROC curve saved to {roc_file}")