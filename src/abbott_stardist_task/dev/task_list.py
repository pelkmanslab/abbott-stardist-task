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
        # Modify the meta according to your task requirements
        # If the task requires a GPU, add "needs_gpu": True
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Segmentation",
        tags=["Instance Segmentation", "Classical segmentation"],
        docs_info="file:docs_info/stardist_segmentation_task.md",
    ),
    
    
    
]
