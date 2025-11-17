"""Pydantic models for advanced iterator configuration."""

from enum import Enum
from typing import Literal, Optional

import numpy as np
from csbdeep.utils import normalize
from ngio import ChannelSelectionModel
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class MaskingConfiguration(BaseModel):
    """Masking configuration.

    Args:
        mode (Literal["Table Name", "Label Name"]): Mode of masking to be applied.
            If "Table Name", the identifier refers to a masking table name.
            If "Label Name", the identifier refers to a label image name.
        identifier (str): Name of the masking table or label image
            depending on the mode.
    """

    mode: Literal["Table Name", "Label Name"] = "Table Name"
    identifier: Optional[str] = None


class IteratorConfiguration(BaseModel):
    """Advanced Masking configuration.

    Args:
        masking (Optional[MaskingIterator]): If configured, the segmentation
            will be only saved within the mask region.
        roi_table (Optional[str]): Name of a ROI table. If provided, the segmentation
            will be performed for each ROI in the specified ROI table.
    """

    masking: Optional[MaskingConfiguration] = Field(
        default=None, title="Masking Iterator Configuration"
    )
    roi_table: Optional[str] = Field(default=None, title="Iterate Over ROIs")


class StardistChannel(BaseModel):
    """Stardist channel configuration.

    Args:
        This model is used to select a channel by label, wavelength ID, or index.

    Args:
        identifiers (str): Unique identifier for the channel.
            This can be a channel label, wavelength ID, or index.
        mode (Literal["label", "wavelength_id", "index"]): Specifies how to
            interpret the identifier. Can be "label", "wavelength_id", or
            "index" (must be an integer). At least one and at most three
            identifiers must be provided.

    """

    mode: Literal["label", "wavelength_id", "index"] = "label"
    identifiers: list[str] = Field(default_factory=list, min_length=1, max_length=2)

    def to_list(self) -> list[ChannelSelectionModel]:
        """Convert to list of ChannelSelectionModel.

        Returns:
            list[ChannelSelectionModel]: List of ChannelSelectionModel.
        """
        return [
            ChannelSelectionModel(identifier=identifier, mode=self.mode)
            for identifier in self.identifiers
        ]


class NormalizationParameters(BaseModel):
    """Validator to handle different normalization scenarios for Stardist models

    If `norm_type="default"`, then Stardist default normalization is
    used and no other parameters can be specified.
    If `norm_type="no_normalization"`, then no normalization is used and no
    other parameters can be specified.
    If `norm_type="custom"`, then either percentiles or explicit integer
    bounds can be applied.

    Attributes:
        norm_type:
            One of `default` (Stardist default normalization), `custom`
            (using the other custom parameters) or `no_normalization`.
        lower_percentile: Specify a custom lower-bound percentile for rescaling
            as a float value between 0 and 100. Set to 1 to run the same as
            default). You can only specify percentiles or bounds, not both.
        upper_percentile: Specify a custom upper-bound percentile for rescaling
            as a float value between 0 and 100. Set to 99 to run the same as
            default, set to e.g. 99.99 if the default rescaling was too harsh.
            You can only specify percentiles or bounds, not both.
        lower_bound: Explicit lower bound value to rescale the image at.
            Needs to be an integer, e.g. 100.
            You can only specify percentiles or bounds, not both.
        upper_bound: Explicit upper bound value to rescale the image at.
            Needs to be an integer, e.g. 2000.
            You can only specify percentiles or bounds, not both.
    """

    norm_type: Literal["default", "custom", "no_normalization"] = "default"
    lower_percentile: Optional[float] = Field(None, ge=0, le=100)
    upper_percentile: Optional[float] = Field(None, ge=0, le=100)
    lower_bound: Optional[int] = None
    upper_bound: Optional[int] = None

    # In the future, add an option to allow using precomputed percentiles
    # that are stored in OME-Zarr histograms and use this pydantic model that
    # those histograms actually exist

    @model_validator(mode="after")
    def validate_conditions(self: Self) -> Self:
        """Validate cross-field conditions after model initialization."""
        # Extract values
        norm_type = self.norm_type
        lower_percentile = self.lower_percentile
        upper_percentile = self.upper_percentile
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        # Verify that custom parameters are only provided when type="custom"
        if norm_type != "custom":
            if lower_percentile is not None:
                raise ValueError(
                    f"Type='{norm_type}' but {lower_percentile=}. "
                    "Hint: set norm_type='custom'."
                )
            if upper_percentile is not None:
                raise ValueError(
                    f"Type='{norm_type}' but {upper_percentile=}. "
                    "Hint: set norm_type='custom'."
                )
            if lower_bound is not None:
                raise ValueError(
                    f"Type='{norm_type}' but {lower_bound=}. "
                    "Hint: set norm_type='custom'."
                )
            if upper_bound is not None:
                raise ValueError(
                    f"Type='{norm_type}' but {upper_bound=}. "
                    "Hint: set norm_type='custom'."
                )

        # The only valid options are:
        # 1. Both percentiles are set and both bounds are unset
        # 2. Both bounds are set and both percentiles are unset
        are_percentiles_set = (
            lower_percentile is not None,
            upper_percentile is not None,
        )
        are_bounds_set = (
            lower_bound is not None,
            upper_bound is not None,
        )
        if len(set(are_percentiles_set)) != 1:
            raise ValueError(
                "Both lower_percentile and upper_percentile must be set together."
            )
        if len(set(are_bounds_set)) != 1:
            raise ValueError("Both lower_bound and upper_bound must be set together")
        if lower_percentile is not None and lower_bound is not None:
            raise ValueError(
                "You cannot set both explicit bounds and percentile bounds "
                "at the same time. Hint: use only one of the two options."
            )

        return self


def normalized_img(
    img: np.ndarray,
    axis: int = -1,
    invert: bool = False,
    lower_p: float = 1.0,
    upper_p: float = 99.0,
    lower_bound: Optional[int] = None,
    upper_bound: Optional[int] = None,
):
    """Normalize each channel of the image so that so that 0.0=lower percentile

    or lower bound and 1.0=upper percentile or upper bound of image intensities.

    The normalization can result in values < 0 or > 1 (no clipping).

    optional inversion

    Parameters
    ------------
    img: ND-array (at least 3 dimensions)
    axis: channel axis to loop over for normalization
    invert: invert image (useful if cells are dark instead of bright)
    lower_p: Lower percentile for rescaling
    upper_p: Upper percentile for rescaling
    lower_bound: Lower fixed-value used for rescaling
    upper_bound: Upper fixed-value used for rescaling

    Returns:
    ---------------
    img: ND-array, float32
        normalized image of same size

    """
    if img.ndim < 3:
        error_message = "Image needs to have at least 3 dimensions"
        raise ValueError(error_message)

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if lower_p is not None:
            # ptp can still give nan's with weird images
            i99 = np.percentile(img[k], upper_p)
            i1 = np.percentile(img[k], lower_p)
            if i99 - i1 > +1e-3:  # np.ptp(img[k]) > 1e-3:
                img[k] = normalize_percentile(img[k], lower=lower_p, upper=upper_p)
                if invert:
                    img[k] = -1 * img[k] + 1
            else:
                img[k] = 0
        elif lower_bound is not None:
            if upper_bound - lower_bound > +1e-3:
                img[k] = normalize_bounds(img[k], lower=lower_bound, upper=upper_bound)
                if invert:
                    img[k] = -1 * img[k] + 1
            else:
                img[k] = 0
        else:
            raise ValueError("No normalization method specified")
    img = np.moveaxis(img, 0, axis)
    return img


def normalize_percentile(Y: np.ndarray, lower: float = 1, upper: float = 99):
    """Normalize image so 0.0 is lower percentile and 1.0 is upper percentile

    Percentiles are passed as floats (must be between 0 and 100)

    Args:
        Y: The image to be normalized
        lower: Lower percentile
        upper: Upper percentile

    """
    X = Y.copy()
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    X = (X - x01) / (x99 - x01)
    return X


def normalize_bounds(Y: np.ndarray, lower: int = 0, upper: int = 65535):
    """Normalize image so 0.0 is lower value and 1.0 is upper value

    Args:
        Y: The image to be normalized
        lower: Lower normalization value
        upper: Upper normalization value

    """
    X = Y.copy()
    X = (X - lower) / (upper - lower)
    return X


def normalize_stardist_channel(
    x: np.ndarray,
    normalization: NormalizationParameters,
) -> np.ndarray:
    """Normalize a Stardist input array by channel.

    Args:
        x: 3D numpy array.
        normalization: By default, data is normalized so 0.0=1st percentile and
            1.0=99th percentile of image intensities in each channel.
            This automatic normalization can lead to issues when the image to
            be segmented is very sparse. You can turn off the default
            rescaling. With the "custom" option, you can either provide your
            own rescaling percentiles or fixed rescaling upper and lower
            bound integers.

    """
    # Optionally perform custom normalization
    if normalization.norm_type == "custom":
        x = normalized_img(
            x,
            lower_p=normalization.lower_percentile,
            upper_p=normalization.upper_percentile,
            lower_bound=normalization.lower_bound,
            upper_bound=normalization.upper_bound,
        )

    if normalization.norm_type == "default":
        x = normalize(x)

    return x


class AdvancedStardistParams(BaseModel):
    """Advanced Stardist Parameters

    Attributes:
        normalization (NormalizationParameters, optional): Normalization parameters.
            The normalization is applied before running the Cellpose model for
            each channel independently.
        sparse: If true, aggregate probabilities/distances sparsely during tiled
            prediction to save memory (recommended)
        prob_thresh: Consider only object candidates from pixels with predicted
            object probability above this threshold.
        nms_thresh: Perform non-maximum suppression (NMS) that
            considers two objects to be the same when their area/surface
            overlap exceeds this threshold.
        scale: Scale the input image internally by a tuple of floats and rescale
            the output accordingly. Useful if the Stardist model has been trained
            on images with different scaling. E.g. (z, y, x) = (1.0, 0.5, 0.5).
        n_tiles : Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled. This parameter denotes a
            tuple of the number of tiles for every image axis.
            E.g. (z, y, x) = (2, 4, 4).
        show_tile_progress: Whether to show progress during tiled prediction.
        verbose: Whether to print some info messages.
        predict_kwargs: Keyword arguments for ``predict`` function of Keras model.
        nms_kwargs: Keyword arguments for non-maximum suppression.
    """

    normalization: NormalizationParameters = NormalizationParameters()
    sparse: bool = True
    prob_thresh: Optional[float] = None
    nms_thresh: Optional[float] = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    n_tiles: tuple[int, int, int] = (1, 1, 1)
    show_tile_progress: bool = False
    verbose: bool = False
    predict_kwargs: dict = None
    nms_kwargs: dict = None


class StardistModels(Enum):
    """Enum for Stardist model names"""

    VERSATILE_FLUO_2D = "2D_versatile_fluo"
    VERSATILE_HE_2D = "2D_versatile_he"
    PAPER_DSB2018_2D = "2D_paper_dsb2018"
    DEMO_2D = "2D_demo"
    DEMO_3D = "3D_demo"


class StardistpretrainedModel(BaseModel):
    """Parameters to load a custom pretrained model

    Attributes:
        base_fld: Base folder to where custom Stardist models are stored
        model_name: Name of the custom model
    """

    base_fld: str
    pretrained_model_name: str
