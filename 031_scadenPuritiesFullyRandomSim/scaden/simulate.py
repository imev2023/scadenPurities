from scaden.simulation import BulkSimulator

"""
Simulation of artificial bulk RNA-seq samples from scRNA-seq data
and subsequenbt formatting in .h5ad file for training with Scaden
Initiate class with BulkSimulator(). If purity is not None, saves the proportion of specified cell type as an additional observation per pseudobulked sample.
"""


def simulation(simulate_dir, 
               data_dir, 
               sample_size, 
               num_samples, 
               pattern,
               unknown_celltypes, 
               out_prefix, # argument for merge_datasets only
               fmt, 
               balance, 
               threshold,
               percRandom,
               saveProp, 
               seed, 
               remMerged, 
               purity, 
               cran):

    unknown_celltypes = list(unknown_celltypes)
    bulk_simulator  = BulkSimulator(sample_size=sample_size,
                                   num_samples=num_samples,
                                   data_path=data_dir,
                                   out_dir=simulate_dir,
                                   pattern=pattern,
                                   unknown_celltypes=unknown_celltypes,
                                   fmt=fmt, 
                                   balance=balance, 
                                   threshold=threshold, 
                                   percRandom=percRandom, 
                                   saveProp=saveProp, 
                                   fprefix=out_prefix, 
                                   seed = seed, 
                                   remMerge=remMerged, 
                                   purity = purity, 
                                   cran=cran)

    # Perform dataset simulation
    bulk_simulator.simulate()

    # Merge the resulting datasets
    # The out_dir and fprefix are called from self.BulkSimulator class so no need to input as arguments.
    bulk_simulator.merge_datasets(files=bulk_simulator.dataset_files)
