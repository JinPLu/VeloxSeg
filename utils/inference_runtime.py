from monai.inferers import sliding_window_inference


def sliding_window_predict(
    inputs,
    predictor,
    roi_size,
    sw_batch_size,
    test_config,
    **kwargs,
):
    return sliding_window_inference(
        inputs,
        roi_size,
        sw_batch_size,
        predictor=predictor,
        overlap=test_config["sliding_window"]["overlap"],
        **kwargs,
    )
