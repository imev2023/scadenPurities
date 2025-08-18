import logging
import glob
import os
import sys
import gc

import pandas as pd
import anndata as ad
import numpy as np
import random

logger = logging.getLogger(__name__)

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
        percRandom=0,
        saveProp=False,
        fprefix = "fprefix",
        seed = None, 
        remMerge = True,
        purity = None, 
        cellRandom = 0
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
        self.percRandom = percRandom
        self.saveProp = saveProp
        self.fprefix = fprefix
        self.seed = seed
        self.remMerge = remMerge
        self.purity = purity
        self.cellRandom = cellRandom
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

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

        prop_records = []
        count_records = []

        for dataset in self.datasets:
            gc.collect()
            logger.info(f"[bold u]Simulating data from {dataset}")
            proportions, counts = self.simulate_dataset(dataset)
            if self.saveProp:
                prop_records.extend(proportions)
                count_records.extend(counts)
                print("Saved simulated proportions and counts.")

        if self.saveProp and prop_records:
            df = pd.DataFrame(prop_records)
            df.to_csv(os.path.join(self.out_dir, self.fprefix + "_bulkSampleProportions.csv"), index=False)

            df_counts = pd.DataFrame(count_records)
            df_counts.to_csv(os.path.join(self.out_dir, self.fprefix + "_bulkSampleCounts.csv"), index=False)


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

        if self.purity and self.purity not in self.celltypes_global:
            logger.error(f"Purity cell type '{self.purity}' not found in cell types: {self.celltypes_global}")
            sys.exit(1)
        else: 
            logger.info(f"Calculating purity of cell type: {self.purity}")
        

        tmp_x, tmp_y, props, counts = self.create_subsample_dataset(data_x, data_y, dataset)

        tmp_x = tmp_x.sort_index(axis=1)
        # ratios = pd.DataFrame(tmp_y, columns=self.celltypes_global)
        ratios = tmp_y.copy()

        ratios["ds"] = pd.Series(np.repeat(dataset, tmp_y.shape[0]), index=ratios.index)

        ann_data = ad.AnnData(
            X=tmp_x.to_numpy(),
            obs=ratios,
            var=pd.DataFrame(columns=[], index=list(tmp_x)),
        )
        ann_data.uns["unknown"] = self.unknown_celltypes
        ann_data.uns["cell_types"] = self.celltypes_global

        ann_data.write(os.path.join(self.out_dir, dataset + ".h5ad"))
        print("Simulated dataset. Saved as " + dataset + ".h5ad" )
        return props, counts
    
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
    
    def create_subsample_dataset(self, x, y, dataset_name):
        sim_x, sim_y = [], []
        prop_records, count_records = [], []

        unique_ids = x["SampleID"].unique().tolist()
        total = self.num_samples
        n_cellrandom = int(self.cellRandom * total)
        n_non_cellrandom = total - n_cellrandom

        n_cr_mixed = int((self.percRandom / 100) * n_cellrandom)
        n_cr_single = n_cellrandom - n_cr_mixed
        n_ncr_mixed = int((self.percRandom / 100) * n_non_cellrandom)
        n_ncr_single = n_non_cellrandom - n_ncr_mixed

        print("--------------------------------------------------------------------")
        print("----------------------------- SIMULATION ---------------------------")
        print("--------------------------------------------------------------------")
        print(f"Generating {self.num_samples} samples:")
        print(f"  - {n_cellrandom} with random cell type proportions:")
        print(f"    - {n_cr_mixed} mixed")
        print(f"    - {n_cr_single} single-sample")
        print(f"  - {n_non_cellrandom} with original proportions:")
        print(f"    - {n_ncr_mixed} mixed")
        print(f"    - {n_ncr_single} single-sample\n")

        def record_sample(sample, label, abs_cts, sample_id, cellrandom_flag):
            sim_x.append(sample)
            sim_y.append(label)
            prop_records.append({
                **dict(zip(self.celltypes_global, label)),
                "SampleID": sample_id,
                "Dataset": dataset_name,
                "Source": "random" if sample_id == "mixed" else "per-sample",
                "cellRandom": cellrandom_flag
            })
            count_records.append({
                **dict(zip(self.celltypes_global, abs_cts)),
                "SampleID": sample_id,
                "Dataset": dataset_name,
                "Source": "random" if sample_id == "mixed" else "per-sample",
                "cellRandom": cellrandom_flag
            })

        # 1. Random proportions, single-sample
        print("--------- Sampling cell random, per sample. ---------")
        for sid in random.choices(unique_ids, k=n_cr_single):
            sub_x = x[x['SampleID'] == sid].drop(columns=['SampleID'])
            sub_y = y[y['SampleID'] == sid].drop(columns=['SampleID'])
            print(sub_x)
            print(sub_y)
            sample, label, abs_cts = self.create_random_subsample(sub_x, sub_y)
            record_sample(sample, label, abs_cts, sid, "Yes")

        # 2. Random proportions, mixed
        print("--------- Sampling cell random, mixed. ---------")
        for _ in range(n_cr_mixed):
            print(x.drop(columns=["SampleID"]))
            print(y.drop(columns=["SampleID"]))
            sample, label, abs_cts = self.create_random_subsample(
                x.drop(columns=["SampleID"]),
                y.drop(columns=["SampleID"])
            )
            record_sample(sample, label, abs_cts, "mixed", "Yes")

        # 3. Original proportions, single-sample
        print("--------- Sampling non-cell random (default), per sample. ---------")
        for sid in random.choices(unique_ids, k=n_ncr_single):
            sub_x = x[x['SampleID'] == sid].drop(columns=['SampleID'])
            sub_y = y[y['SampleID'] == sid].drop(columns=['SampleID'])
            print(sub_x)
            print(sub_y)
            sample, label, abs_cts = self.create_balanced_subsample(sub_x, sub_y)
            record_sample(sample, label, abs_cts, sid, "No")

        # 4. Original proportions, mixed
        print("--------- Sampling non-cell random (default), mixed. ---------")
        for _ in range(n_ncr_mixed):
            print(x.drop(columns=["SampleID"]))
            print(y.drop(columns=["SampleID"]))
            sample, label, abs_cts = self.create_balanced_subsample(
                x.drop(columns=["SampleID"]),
                y.drop(columns=["SampleID"])
            )
            record_sample(sample, label, abs_cts, "mixed", "No")

        # Combine
        x_final = pd.DataFrame(sim_x)
        y_final = pd.DataFrame(sim_y)
        y_final["SampleID"] = [r["SampleID"] for r in prop_records]

        proportions = pd.DataFrame(prop_records)
        abs_counts = pd.DataFrame(count_records)

        print("--------- Sampling cell random, across different samples. ---------") 
        return x_final, y_final, proportions, abs_counts

    def create_random_subsample(self, x, y):
        fractions = self.create_fractions()
        return self._generate_sample(x, y, fractions)

    def create_balanced_subsample(self, x, y):
        fractions = self.create_balanced_fractions(x, y)
        return self._generate_sample(x, y, fractions)

    def _generate_sample(self, x, y, fractions):
        cell_counts = np.round(np.array(fractions) * self.sample_size).astype(int)
        print(f"cell_counts: {cell_counts}")
        sample_cells = []
        sample_labels = []
        abs_cts = []

        for idx, celltype in enumerate(self.celltypes_global):
            print(f"idx: {idx}, celltype: {celltype}")
            n_cells = cell_counts[idx]
            if n_cells > 0:
                # ⚠️ Assume y is a Series of labels (not one-hot)
                subset_x = x[y == celltype]
                if len(subset_x) == 0:
                    raise ValueError(f"No cells of type '{celltype}' found in input data.")
                sampled = subset_x.sample(n=n_cells, replace=True)
                sample_cells.append(sampled)
                sample_labels.extend([fractions[idx]] * n_cells)
                abs_cts.append(n_cells)
            else:
                abs_cts.append(0)

        sample_df = pd.concat(sample_cells) if sample_cells else pd.DataFrame(columns=x.columns)
        return sample_df.reset_index(drop=True), sample_labels, abs_cts

    def create_fractions(self):
        # Random proportions for each cell type
        fractions = np.random.uniform(self.prop_range[0], self.prop_range[1], len(self.celltypes_global))
        fractions /= fractions.sum()  # Normalize to sum to 1
        return fractions

    def create_balanced_fractions(self, x, y):
        freqs = []
        for ct in self.celltypes_global:
            if ct in y.columns:
                ct_count = y[ct].sum()
                freqs.append(ct_count)
            else:
                freqs.append(0)
        freqs = np.array(freqs, dtype=float)
        if freqs.sum() == 0:
            # Avoid divide-by-zero
            freqs = np.ones(len(freqs))
        return freqs / freqs.sum()

    def merge_datasets(self, 
                       files=None):
        """
        Merges .h5ad files and optionally removes them after merging.

        @param files: list of files to merge
        @param remMerged: if True, delete input files after merging
        @return: None
        """
        import os
        import glob
        import anndata as ad
        import logging

        logger = logging.getLogger(__name__)
        non_celltype_obs = ["ds", "batch", "purity"]

        if not files:
            files = glob.glob(os.path.join(self.data_dir, "*.h5ad"))

        logger.info(f"Merging datasets: {files} into [bold cyan]{self.fprefix}")

        # load first file
        adata = ad.read_h5ad(files[0])

        for i in range(1, len(files)):
            adata = adata.concatenate(ad.read_h5ad(files[i]), uns_merge="same")

        combined_celltypes = list(adata.obs.columns)
        combined_celltypes = [x for x in combined_celltypes if x not in non_celltype_obs]
        for ct in combined_celltypes:
            adata.obs[ct].fillna(0, inplace=True)

        adata.uns["cell_types"] = combined_celltypes
        merged_file = os.path.join(self.out_dir, f"{self.fprefix}.h5ad")
        adata.write(merged_file)

        print(f"Merged and saved dataset. Saved as {merged_file}")

        if self.remMerge:
            for file in files:
                try:
                    os.remove(file)
                    logger.info(f"Removed unmerged file: {file}")
                    print(f"Removed unmerged file: {file}")
                except Exception as e:
                    logger.warning(f"Failed to remove unmerged file {file}: {e}")
                    print(f"Failed to remove unmerged file {file}: {e}")
