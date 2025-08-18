import logging
import glob
import os
import sys
import gc

import pandas as pd
import anndata as ad
import numpy as np

from rich.progress import BarColumn, Progress

logger = logging.getLogger(__name__)


def create_fractions(no_celltypes):
    fracs = np.random.rand(no_celltypes)
    return fracs / np.sum(fracs)


class BulkSimulator(object):
    def __init__(
        self,
        sample_size=100,
        num_samples=1000,
        data_path="./",
        out_dir="./",
        pattern="*_counts.txt",
        unknown_celltypes=None,
        fmt="txt",
        balance=False,
        threshold=None,
    ):
        if unknown_celltypes is None:
            unknown_celltypes = ["unknown"]

        self.sample_size = sample_size
        self.num_samples = num_samples
        self.data_path = data_path
        self.out_dir = out_dir
        self.pattern = pattern
        self.unknown_celltypes = unknown_celltypes
        self.format = fmt
        self.datasets = []
        self.dataset_files = []
        self.balance = balance
        self.threshold = threshold
        self.celltypes_global = []

    def simulate(self):
        if not self.data_path.endswith("/"):
            self.data_path += "/"
        files = glob.glob(os.path.join(self.data_path, self.pattern))
        files = [os.path.basename(x) for x in files]
        self.datasets = [x.replace(self.pattern.replace("*", ""), "") for x in files]
        self.dataset_files = [os.path.join(self.out_dir, x + ".h5ad") for x in self.datasets]

        if len(self.datasets) == 0:
            logging.error("No datasets found! Have you specified the pattern correctly?")
            sys.exit(1)

        logger.info("Datasets: [cyan]" + str(self.datasets) + "[/]")

        for dataset in self.datasets:
            gc.collect()
            logger.info(f"[bold u]Simulating data from {dataset}")
            self.simulate_dataset(dataset)

        logger.info("[bold green]Finished data simulation!")

    def simulate_dataset(self, dataset):
        data_x, data_y = self.load_dataset(dataset)

        sample_map_file = os.path.join(self.data_path, dataset + "_samples.txt")
        sample_map = pd.read_table(sample_map_file, header=None)
        sample_map.columns = ['SampleID']
        data_y['SampleID'] = sample_map['SampleID']
        data_x['SampleID'] = sample_map['SampleID']

        logger.info(f"Merging unknown cell types: {self.unknown_celltypes}")
        data_y = self.merge_unknown_celltypes(data_y)

        logger.info(f"Subsampling [bold cyan]{dataset}[/] ...")

        self.celltypes_global = sorted(set(data_y['Celltype']))
        tmp_x, tmp_y = self.create_subsample_dataset(data_x, data_y)

        tmp_x = tmp_x.sort_index(axis=1)
        ratios = pd.DataFrame(tmp_y, columns=self.celltypes_global)
        ratios["ds"] = pd.Series(np.repeat(dataset, tmp_y.shape[0]), index=ratios.index)

        ann_data = ad.AnnData(
            X=tmp_x.to_numpy(),
            obs=ratios,
            var=pd.DataFrame(columns=[], index=list(tmp_x)),
        )
        ann_data.uns["unknown"] = self.unknown_celltypes
        ann_data.uns["cell_types"] = self.celltypes_global

        ann_data.write(os.path.join(self.out_dir, dataset + ".h5ad"))

    def load_dataset(self, dataset):
        pattern = self.pattern.replace("*", "")
        logger.info(f"Loading [cyan]{dataset}[/] dataset ...")
        dataset_counts = dataset + pattern
        dataset_celltypes = dataset + "_celltypes.txt"

        if self.format == "txt":
            y = pd.read_table(os.path.join(self.data_path, dataset_celltypes))
            if "Celltype" not in y.columns:
                logger.error(f"No 'Celltype' column found in {dataset}_celltypes.txt!")
                sys.exit()
            x = pd.read_table(
                os.path.join(self.data_path, dataset_counts), index_col=0, dtype=np.float32
            )
            if not y.shape[0] == x.shape[0]:
                logger.error("Mismatch between celltypes and counts files!")
                sys.exit(1)
        elif self.format == "h5ad":
            data_h5ad = ad.read_h5ad(os.path.join(self.data_path, dataset_counts))
            y = pd.DataFrame(data_h5ad.obs.Celltype)
            y.reset_index(inplace=True, drop=True)
            x = pd.DataFrame(data_h5ad.X.todense())
            x.index = data_h5ad.obs_names
            x.columns = data_h5ad.var_names
            del data_h5ad
        else:
            logger.error(f"Unsupported file format {self.format}!")
            sys.exit(1)

        return x, y

    def merge_unknown_celltypes(self, y):
        y["Celltype"] = ["Unknown" if x in self.unknown_celltypes else x for x in y["Celltype"]]
        return y

    def create_subsample_dataset(self, x, y):
        sim_x, sim_y = [], []
        unique_ids = y['SampleID'].unique()
        samples_per_id = self.num_samples // len(unique_ids)

        for sid in unique_ids:
            sub_x = x[x['SampleID'] == sid].drop(columns=['SampleID'])
            sub_y = y[y['SampleID'] == sid].drop(columns=['SampleID'])
            sid_celltypes = sub_y['Celltype'].unique().tolist()

            for _ in range(samples_per_id):
                sample, label = self.create_balanced_subsample(sub_x, sub_y, sid_celltypes)
                sim_x.append(sample)
                sim_y.append(label)

        sim_x = pd.concat(sim_x, axis=1).T
        sim_y = pd.DataFrame(sim_y, columns=self.celltypes_global)

        return sim_x, sim_y

    def create_balanced_subsample(self, x, y, available_celltypes):
        if self.balance:
            count = self.sample_size // len(available_celltypes)
            counts = [count] * len(available_celltypes)
            for i in np.random.choice(len(available_celltypes), self.sample_size % len(available_celltypes), replace=False):
                counts[i] += 1
        else:
            if self.threshold is None:
                raise ValueError("Threshold must be set when balance is False")

            threshold_cts = int((self.threshold / 100.0) * self.sample_size)
            total_assigned = threshold_cts * len(available_celltypes)
            remainder = self.sample_size - total_assigned

            ct_counts = y['Celltype'].value_counts(normalize=True).to_dict()
            proportions = np.array([ct_counts.get(ct, 0) for ct in available_celltypes])
            norm_proportions = proportions / proportions.sum()
            additional_counts = (norm_proportions * remainder).astype(int)
            counts = [threshold_cts + extra for extra in additional_counts]

        sample_cells = []
        fracs_complete = [0] * len(self.celltypes_global)
        for i, (ct, n_cells) in enumerate(zip(available_celltypes, counts)):
            ct_cells = x[y['Celltype'] == ct]
            if len(ct_cells) < n_cells:
                sampled_cells = ct_cells.sample(n=len(ct_cells), replace=True)
            else:
                sampled_cells = ct_cells.sample(n=n_cells, replace=False)
            sample_cells.append(sampled_cells)
            if ct in self.celltypes_global:
                fracs_complete[self.celltypes_global.index(ct)] = n_cells / self.sample_size

        df_samp = pd.concat(sample_cells, axis=0).sum(axis=0)
        return df_samp, fracs_complete
    
    @staticmethod
    def merge_datasets(data_dir="./", 
                       files=None, 
                       out_name="scadenSimulateMergeOutput"):
        """

        @param out_name: name of the merged .h5ad file
        @param data_dir: directory to look for datasets
        @param files: list of files to merge
        @return:
        """
        non_celltype_obs = ["ds", "batch"]
        if not files:
            files = glob.glob(os.path.join(data_dir, "*.h5ad"))

        logger.info(f"Merging datasets: {files} into [bold cyan]{out_name}")

        # load first file
        adata = ad.read_h5ad(files[0])

        for i in range(1, len(files)):
            adata = adata.concatenate(ad.read_h5ad(files[i]), uns_merge="same")

        combined_celltypes = list(adata.obs.columns)
        combined_celltypes = [
            x for x in combined_celltypes if not x in non_celltype_obs
        ]
        for ct in combined_celltypes:
            adata.obs[ct].fillna(0, inplace=True)

        adata.uns["cell_types"] = combined_celltypes
        adata.write(out_dir + out_name + ".h5ad")
