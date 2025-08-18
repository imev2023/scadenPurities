from concurrent.futures import ProcessPoolExecutor
from testingPurityFunctions.sharedConfig import generate_batches, BULK, BULK_PURITY, PURITY
from logBatchesNew import load_completed_batches, mark_batch_completed
import scaden
print(scaden.__file__)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from scaden.simulate import simulation
from scaden.process import processing
from scaden.train import training
from scaden.predict import prediction



import os, shutil, gc

# Idea is to collect param_grid options for all seeds (i.e. 5 at a time). Then run simulate and process with parallel computing. Training is very internsive so running training one at a time once the data is processed. 

def get_batch_id(batch):
    subDir, bal, thr, ran, var, sam, cel, _, lr, steps, cran = batch[0]
    name = os.path.basename(subDir.rstrip("/"))
    return f"{name}_bal{bal}_thresh{thr}_ran{ran}_var{var}_samp{sam}_cells{cel}_lr{lr}_steps{steps}_cran{cran}"


def simulate_one(args):
    subDir, bal, thr, ran, var, sam, cel, seed, lr, steps, cran = args
    pfix = f"{os.path.basename(subDir.rstrip('/'))}_bal{bal}_thresh{thr}_ran{ran}_var{var}_samp{sam}_cells{cel}_seed{seed}_lr{lr}_steps{steps}_cran{cran}"
    simDir = f"/Users/nv4/gitClones/scadenPuritiesFullyRandomSim/DFT1/results/1_simulate/{pfix}/"
    
    os.makedirs(simDir, exist_ok=True)
    simulation(
        simulate_dir=simDir,
        data_dir=subDir,
        sample_size=cel,
        num_samples=sam,
        pattern="*_counts.txt",
        unknown_celltypes=["unknown"],
        out_prefix=pfix,
        fmt="txt",
        balance=bal,
        threshold=thr,
        percRandom=ran,
        saveProp=True,
        seed=seed,
        remMerged=True,
        purity = PURITY
    )

def process_one(args, purity_col= "purity"):
    subDir, bal, thr, ran, var, sam, cel, seed, lr, steps, cran = args
    pfix = f"{os.path.basename(subDir.rstrip('/'))}_bal{bal}_thresh{thr}_ran{ran}_var{var}_samp{sam}_cells{cel}_seed{seed}_lr{lr}_steps{steps}_cran{cran}"
    simDir = f"/Users/nv4/gitClones/scadenPuritiesFullyRandomSim/DFT1/results/1_simulate/{pfix}/"
    procDir = f"/Users/nv4/gitClones/scadenPuritiesFullyRandomSim/DFT1/results/2_process/{pfix}/"
    os.makedirs(procDir, exist_ok=True)

    training_data = os.path.join(simDir, pfix + ".h5ad")
    processed_path = os.path.join(procDir, pfix + "_Processed.h5ad")

    processing(
        data_path=BULK,
        training_data=training_data,
        processed_path=processed_path,
        var_cutoff=var ,
        ignore_genes=None, # Helpful if we need to ignore
        purity_col = purity_col#,
        # purity_bin_size=10, # Default. Should only bin IF there is a purity column in the simulated data. 
        # ignore_genes="purity" # Because this hasn't been set as a "gene" for the simulated data yet, we need to ignore it from the bulk matrix. Will adjust this so that it is done in this function iso in R before. 
    )

    return processed_path, training_data

def train_one(args, processed_path, purity_col = "purity"):
    subDir, bal, thr, ran, var, sam, cel, seed, lr, steps, cran = args
    pfix = f"{os.path.basename(subDir.rstrip('/'))}_bal{bal}_thresh{thr}_ran{ran}_var{var}_samp{sam}_cells{cel}_seed{seed}_lr{lr}_steps{steps}_cran{cran}"
    modDir = f"/Users/nv4/gitClones/scadenPuritiesFullyRandomSim/DFT1/results/3_train/{pfix}/"
    metDir = f"/Users/nv4/gitClones/scadenPuritiesFullyRandomSim/DFT1/results/3_metrics/{pfix}/"
    os.makedirs(modDir, exist_ok=True)
    os.makedirs(metDir, exist_ok=True)

    training(
        train_datasets="",
        data_path=processed_path,
        seed=seed,
        batch_size=128,
        learning_rate=lr,
        num_steps=steps,
        model_dir=modDir,
        metric_dir=metDir,
        fprefix=pfix, 
        purity_col = purity_col
    )

def predict_one(args, purity_col = "purity"):
    subDir, bal, thr, ran, var, sam, cel, seed, lr, steps, cran = args
    pfix = f"{os.path.basename(subDir.rstrip('/'))}_bal{bal}_thresh{thr}_ran{ran}_var{var}_samp{sam}_cells{cel}_seed{seed}_lr{lr}_steps{steps}_cran{cran}"
    modDir = f"/Users/nv4/gitClones/scadenPuritiesFullyRandomSim/DFT1/results/3_train/{pfix}/"
    predDir = f"/Users/nv4/gitClones/scadenPuritiesFullyRandomSim/DFT1/results/4_predict/{pfix}/"
    os.makedirs(predDir, exist_ok=True)

    prediction(
        model_dir=modDir,
        data_path=BULK_PURITY,
        out_name=os.path.join(predDir, f"{pfix}_Predicted.txt"),
        purity_col=purity_col
    )

    # Cleanup
    if os.path.exists(modDir):
        shutil.rmtree(modDir)
    gc.collect()


if __name__ == "__main__":
    batches = generate_batches()
    completed = load_completed_batches()

    for batch in batches:
        batch_id = get_batch_id(batch)
        if batch_id in completed:
            print(f"Skipping already completed batch: {batch_id}")
            continue

        print(f"=== Starting batch {batch_id} ===")

        # --- Simulate (parallel: max_workers=2)
        
        print(f"=== Simulating. ===")
        with ProcessPoolExecutor(max_workers=2) as sim_pool:
            sim_pool.map(simulate_one, batch)

        # --- Process (parallel: max_workers=2)
        print(f"=== Processing. ===")
        
        with ProcessPoolExecutor(max_workers=2) as proc_pool:
            processed_results = list(proc_pool.map(process_one, batch))

        # --- Train and Predict (sequential, max_workers=1 logic)
        print(f"=== Train and predict.. ===")
        for args, (processed_path, training_data) in zip(batch, processed_results):
            train_one(args, processed_path)
            predict_one(args)

            # Cleanup memory
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if os.path.exists(training_data):
                os.remove(training_data)

        # --- Mark completed
        mark_batch_completed(batch_id)
        print(f"=== Completed batch {batch_id} ===\n")
