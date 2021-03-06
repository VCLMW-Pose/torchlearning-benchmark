from engine.data import datasets

from .mnist_eval import do_mnist_evaluation
from .mwpose_eval import do_mwpose_evaluation
from .modelnet_eval import do_modelnet_evaluation


def evaluate(dataset, predictions, gts, output_folder):
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
        predictions=predictions, gts=gts, output_folder=output_folder,
    )
    if isinstance(dataset, datasets.MNIST):
        return do_mnist_evaluation(**args)
    elif isinstance(dataset, datasets.MWPose):
        return do_mwpose_evaluation(dataset=dataset, **args)
    elif isinstance(dataset, datasets.ModelNetHdf):
        return do_modelnet_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
