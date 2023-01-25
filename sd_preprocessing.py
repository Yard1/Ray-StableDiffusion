import random
from functools import partial
from typing import Any, Dict

import datasets
import numpy as np
import pandas as pd
import ray
import ray.data
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors import BatchMapper, Chain, TorchVisionPreprocessor
from torchvision import transforms
from transformers import CLIPTokenizer

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def get_image_and_caption_columns(args, dataset_name_mapping, column_names):
    # Get the column names for input/target.
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    return image_column, caption_column


class Tokenizer:
    def __init__(self, pretrained_model_name_or_path, revision, caption_column) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
        )
        self.caption_column = caption_column

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(self, examples, is_train=True):
        captions = []
        for caption in examples[self.caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{self.caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return inputs.input_ids

    def __call__(self, df: "pd.DataFrame") -> "pd.DataFrame":
        df["input_ids"] = list(self.tokenize_captions(df))
        df = df.drop(self.caption_column, axis=1)
        return df


class TokenizerPreprocessor(Preprocessor):
    _is_fittable = False

    def __init__(self, pretrained_model_name_or_path, revision, caption_column) -> None:
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.revision = revision
        self.caption_column = caption_column

    _transform_pandas = Tokenizer

    def _get_transform_config(self) -> Dict[str, Any]:
        """Returns kwargs to be passed to :meth:`ray.data.Dataset.map_batches`.

        This can be implemented by subclassing preprocessors.
        """
        return dict(
            compute=ray.data.ActorPoolStrategy(),
            fn_constructor_kwargs=dict(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                revision=self.revision,
                caption_column=self.caption_column,
            ),
        )


def ensure_correct_format(batch: pd.DataFrame, image_column) -> pd.DataFrame:
    # Converts image to RGB and removes unnecessary column
    image_deature = datasets.Image()
    batch["pixel_values"] = [
        image_deature.decode_example(image).convert("RGB")
        for image in batch[image_column]
    ]
    batch = batch.drop(image_column, axis=1)
    return batch


def get_preprocessor(args, dataset: ray.data.Dataset) -> Preprocessor:
    column_names = dataset.schema(fetch_if_missing=True).names

    # Get the column names for input/target.
    image_column, caption_column = get_image_and_caption_columns(
        args, DATASET_NAME_MAPPING, column_names
    )

    torchvision_transform = transforms.Compose(
        [
            transforms.Resize(
                args.resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    tokenizer = TokenizerPreprocessor(
        args.pretrained_model_name_or_path, args.revision, caption_column
    )
    image_preprocessor = TorchVisionPreprocessor(
        ["pixel_values"],
        torchvision_transform,
    )
    ensure_correct_format_preprocessor = BatchMapper(
        partial(ensure_correct_format, image_column=image_column),
        batch_format="pandas",
    )
    return Chain(tokenizer, ensure_correct_format_preprocessor, image_preprocessor)
