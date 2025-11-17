"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import (
    ParallelTask,
)

AUTHORS = "Ruth Hornbachner"


DOCS_LINK = "https://github.com/pelkmanslab/abbott-stardist-task"


INPUT_MODELS = [
    ("ngio", "images/_image.py", "ChannelSelectionModel"),
    (
        "abbott_stardist_task",
        "utils.py",
        "MaskingConfiguration",
    ),
    (
        "abbott_stardist_task",
        "utils.py",
        "IteratorConfiguration",
    ),
]

TASK_LIST = [
    ParallelTask(
        name="Stardist Segmentation",
        executable="stardist_segmentation_task.py",
        meta={"cpus_per_task": 4, "mem": 16000, "needs_gpu": True},
        category="Segmentation",
        tags=["Instance Segmentation", "Classical segmentation", "2D", "3D"],
        docs_info="file:docs_info/stardist_segmentation_task.md",
    ),
]
