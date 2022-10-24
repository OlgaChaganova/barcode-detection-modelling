import argparse
import logging
import os
import random
import shutil
import typing as tp
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def parse() -> tp.Any:
    parser = argparse.ArgumentParser(description='Parser for data convertation to YoloV5 format script.')
    parser.add_argument(
        '--images_path',
        type=str,
        default='data/barcodes-annotated-gorai/images',
        help='Path to folder images.',
    )
    parser.add_argument(
        '--annot_path',
        type=str,
        default='data/barcodes-annotated-gorai/full_annotation.tsv',
        help='Path to full annotations.',
    )
    parser.add_argument(
        '--new_root',
        type=str,
        default='data/barcodes-annotated-gorai_prepared/',
        help='Path to new root that will contain images and labels.',
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.1,
        help='Share of test images.',
    )
    return parser.parse_args()


def copy_files(images_list: tp.List[str], dest_folder: str) -> None:
    """
    Copy files.

    Parameters
    ----------
    images_list : tp.List[str]
        List with paths to images to be copied.

    dest_folder : str
        Destination folder for copied images.
    """
    for img_path in tqdm(images_list):
        file_name = os.path.split(img_path)[-1]
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copy(
            src=img_path,
            dst=dest_path,
        )


class DatasetPreparator(object):
    """Dataset preparation class."""

    def __init__(
        self,
        images_path: str,
        annotations_path: str,
        new_root: str,
        test_size: float,
        file_ext: str = 'jpg',
    ):
        """
        Initialize DatasetPreparator.

        Parameters
        ----------
        images_path : str
            Path to root directory with all images.

        annotations_path : str
            Path to labels of the original dataset.

        new_root : str
            Path to new root directory with prepared images and annotations.

        test_size : float
            Share of validation and test data.

        file_ext : str
            Image file extension.
        """
        self._images_path = images_path
        self._new_root = new_root
        self._annot_df = pd.read_csv(annotations_path, sep='\t')
        self._test_size = test_size
        self._file_ext = file_ext

    def prepare(self):
        train_inds, val_inds, test_inds = self._train_test_val_split()
        train_images, val_images, test_images = self._split_images(train_inds, val_inds, test_inds)

        self._prepare_images(train_images, val_images, test_images)
        self._prepare_annotations(train_inds, val_inds, test_inds)

    def _train_test_val_split(self) -> tp.Tuple[tp.List[int], tp.List[int], tp.List[int]]:
        """
        Train / val / test split.

        Returns
        -------
            Indexes of train, val, test subsets.
        """
        indexes = list(self._annot_df.index)
        random.shuffle(indexes)
        train_cnt = int(len(self._annot_df) * (1 - self._test_size))
        test_val_cnt = (len(self._annot_df) - train_cnt) // 2
        train_inds = indexes[:train_cnt]
        val_inds = indexes[train_cnt:train_cnt + test_val_cnt]
        test_inds = indexes[train_cnt + test_val_cnt:]
        return train_inds, val_inds, test_inds

    def _split_images(
        self,
        train_inds: tp.List[int],
        val_inds: tp.List[int],
        test_inds: tp.List[int],
    ) -> tp.Tuple[tp.List[int], tp.List[int], tp.List[int]]:
        """
        Split images by the train / val / test inds.

        Parameters
        ----------
        train_inds: tp.List[int]
            Training indices.

        val_inds: tp.List[int]
            Validation indices.

        test_inds: tp.List[int]
            Test indices.

        Returns
        -------
            Lists with path to train / val / test images
        """
        transform = lambda row: os.path.join(self._images_path, row)  # noqa: E731
        train_images = self._annot_df.loc[train_inds, 'filename'].apply(transform)
        val_images = self._annot_df.loc[val_inds, 'filename'].apply(transform)
        test_images = self._annot_df.loc[test_inds, 'filename'].apply(transform)
        logging.info(f'Train images: {len(train_images)}')
        logging.info(f'Val images: {len(val_images)}')
        logging.info(f'Test images: {len(test_images)}')
        return train_images.tolist(), val_images.tolist(), test_images.tolist()

    def _prepare_images(
        self,
        train_images: tp.List[Path],
        val_images: tp.List[Path],
        test_images: tp.List[Path],
    ) -> None:
        """
        Create train, val, test sub folders in the root folder and copy images there.

        Parameters
        ----------
        train_images : tp.List[Path]
            List with paths to train images.

        val_images : tp.List[Path]
            List with paths to val images.

        test_images : tp.List[Path]
            List with paths to test images.
        """
        new_images_path = os.path.join(self._new_root, 'images')
        train_path = os.path.join(new_images_path, 'train')
        val_path = os.path.join(new_images_path, 'val')
        test_path = os.path.join(new_images_path, 'test')

        Path(new_images_path).mkdir(parents=True, exist_ok=True)
        Path(train_path).mkdir(parents=True, exist_ok=True)
        Path(val_path).mkdir(parents=True, exist_ok=True)
        Path(test_path).mkdir(parents=True, exist_ok=True)

        for images_list, dest_folder in zip([train_images, val_images, test_images], [train_path, val_path, test_path]):
            logging.info(f'Writing to {dest_folder}')
            copy_files(images_list=images_list, dest_folder=dest_folder)

    def _convert_annotation(self, index: int) -> tp.Tuple[float, float, float, float]:
        """
        Convert annotations for the yolov5 format.

        Parameters
        ----------
        index : int
            Index of the dataframe row.

        Returns
        -------
        X central, Y central, height and width of the bounding box.
        """
        img_row = self._annot_df.iloc[index, :]
        image_path = os.path.join(self._images_path, img_row.filename)
        image = cv2.imread(image_path)

        p1_p2 = (img_row.p1, img_row.p2)
        p1_p2 = [point.replace('(', '').replace(')', '').split(',') for point in p1_p2]
        p1, p2 = [(int(point[0]), int(point[1])) for point in p1_p2]
        y_min, x_min = p1
        y_max, x_max = p2
        h_abs, w_abs, _ = image.shape

        xc = (x_min + x_max) / 2 / w_abs
        yc = (y_min + y_max) / 2 / h_abs
        h_rel = (y_max - y_min) / h_abs
        w_rel = (x_max - x_min) / w_abs
        return xc, yc, w_rel, h_rel

    def _prepare_annotations(
        self,
        train_inds: tp.List[int],
        val_inds: tp.List[int],
        test_inds: tp.List[int],
    ) -> None:
        new_labels_path = os.path.join(self._new_root, 'labels')
        Path(new_labels_path).mkdir(parents=True, exist_ok=True)

        train_df = self._annot_df.loc[train_inds, :]
        val_df = self._annot_df.loc[val_inds, :]
        test_df = self._annot_df.loc[test_inds, :]

        for df, mode in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
            df_labels_path = os.path.join(new_labels_path, mode)
            logging.info(f'Writing to {df_labels_path}')
            Path(df_labels_path).mkdir(parents=True, exist_ok=True)
            for index in tqdm(df.index):
                xc, yc, h_rel, w_rel = self._convert_annotation(index)
                num_class = 0
                txt = f'{num_class} {xc} {yc} {h_rel} {w_rel}'
                image_filename = df.loc[index, 'filename']
                annot_filename = os.path.split(image_filename)[-1]
                annot_filename = annot_filename.replace(self._file_ext, 'txt')
                annot_path = os.path.join(df_labels_path, annot_filename)
                with open(annot_path, 'w') as annot_file:
                    annot_file.write(txt)


if __name__ == '__main__':
    args = parse()

    logging.basicConfig(level=logging.INFO)

    dataset_preparator = DatasetPreparator(
        images_path=args.images_path,
        annotations_path=args.annot_path,
        new_root=args.new_root,
        test_size=args.test_size,
    )

    dataset_preparator.prepare()
