from engine.data import datasets

# from .mnist_eval import do_mnist_evaluation
from .mwpose_save import do_mwpose_visualization


def torchlearning_save(dataset, predictions, output_folder):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(dict): each item in the list represents the
            prediction results for one image.
        gt(dict): Ground truth for each batch
        output_folder: output folder, to save evaluation files or results.
    Returns:
        evaluation result
    """
    args = dict(
        predictions=predictions, output_folder=output_folder,
    )
    if isinstance(dataset, datasets.MNIST):
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
        # return do_mnist_evaluation(**args)
    elif isinstance(dataset, datasets.MWPose):
        return do_mwpose_visualization(dataset=dataset, **args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
