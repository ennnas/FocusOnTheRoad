from typing import Sequence, Tuple

import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    num_images: int = 20,
    val_images: int = 5,
    group_ids: Sequence[str] = ("subject", "classname"),
    seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped_df = df.groupby(group_ids)
    # consider only groups with at least NUM_IMAGES images
    grouped_df = grouped_df.filter(lambda x: x.img.count() > num_images)
    sampled_df = (
        grouped_df.groupby(group_ids)
        .apply(lambda x: x.sample(num_images, random_state=seed))
        .reset_index(drop=True)
    )
    test_df = (
        sampled_df.groupby(group_ids)
        .apply(lambda x: x.sample(val_images, random_state=seed))
        .reset_index(drop=True)
    )
    train_df = pd.concat([sampled_df, test_df]).drop_duplicates(keep=False)
    assert set(train_df.img.to_list()) & set(test_df.img.to_list()) == set()
    return train_df, test_df
