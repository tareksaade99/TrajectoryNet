import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

# TrajectoryNet Imports
from TrajectoryNet import dataset, eval_utils
from TrajectoryNet.parse import parser
from TrajectoryNet.lib.growth_net import GrowthNet # Keep if you might load a growth model
from TrajectoryNet.lib.viz_scrna import trajectory_to_video, save_vectors
from TrajectoryNet.lib.viz_scrna import (
    save_trajectory_density,
    save_2d_trajectory,
    save_2d_trajectory_v2,
)
from TrajectoryNet.train_misc import (
    set_cnf_options,
    build_model_tabular,
    create_regularization_fns,
    add_spectral_norm, # Keep if you used it during training
)

# --- Keep necessary helper/visualization functions if you want to use them ---
# (You can keep makedirs, plot_output, etc., but we'll focus the main part on eval)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def plot_output(device, args, model, data):
    """
    Generates and saves density plots and videos.
    Adjust parameters as needed.
    """
    print("    Generating density plots...")
    data_samples = data.get_data()[data.sample_index(2000, 0)] # Sample data
    density_dir = os.path.join(args.save, "density_eval")
    makedirs(density_dir) # Ensure directory exists
    
    try:
        save_trajectory_density(
            data.base_density(),
            model,
            data_samples,
            density_dir,
            device=device,
            end_times=args.int_tps,
            ntimes=50, # Reduced ntimes for potentially faster eval
            memory=0.5, # Adjusted memory for eval
        )
        print(f"    Density plots saved in {density_dir}. Creating video...")
        trajectory_to_video(density_dir)
        print("    Density video created.")
    except Exception as e:
        print(f"    Could not generate density plots/video: {e}")

# --- Main Evaluation Script ---

def main(args):
    # --- 1. Setup ---
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() and not args.use_cpu else "cpu"
    )
    print(f"Using device: {device}")

    print(f"Loading dataset: {args.dataset}")
    data = dataset.SCData.factory(args.dataset, args)
    args.data = data # Attach data to args for eval_utils

    args.timepoints = data.get_unique_times()
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale
    print(f"Timepoints: {args.timepoints}")
    print(f"Integration times: {args.int_tps}")

    # --- 2. Build Model ---
    print("Building model...")
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, data.get_shape()[0], regularization_fns).to(
        device
    )
    
    # --- 3. Load Growth Model (If Used) ---
    growth_model = None
    if args.use_growth:
        try:
            growth_model_path = data.get_growth_net_path()
            print(f"Loading growth model from: {growth_model_path}")
            growth_model = torch.load(growth_model_path, map_location=device)
            growth_model.eval() # Set to evaluation mode
        except Exception as e:
            print(f"Warning: Could not load growth model: {e}")
            growth_model = None # Ensure it's None if loading fails

    # --- 4. Set Options and Load Checkpoint ---
    if args.spectral_norm: # Only add if used during training
         add_spectral_norm(model)
    set_cnf_options(args, model)

    checkpoint_path = os.path.join(args.save, "checkpt.pth")
    print(f"Loading trained model checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}!")
        print("Ensure your --save argument points to the correct directory.")
        return # Exit if checkpoint not found

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict["state_dict"])
    
    # Check if growth model state was saved and load it if needed
    if args.use_growth and growth_model is not None and "growth_state_dict" in state_dict:
        print("Loading growth model state from checkpoint.")
        growth_model.load_state_dict(state_dict["growth_state_dict"])
        growth_model.eval()
    elif args.use_growth:
         print("Warning: --use_growth is set, but no growth model state found/loaded.")


    model.eval() # <--- Set model to evaluation mode! VERY IMPORTANT!
    print("Model loaded and set to evaluation mode.")

    # --- 5. Run Evaluations ---
    print("\n--- Starting Model Evaluation ---")
    
    # A. Negative Log Likelihood (NLL)
    print("\n[1] Calculating Negative Log Likelihood (NLL)...")
    try:
        nll = eval_utils.evaluate(device, args, model, growth_model)
        print(f"    NLL per timepoint saved to {args.save}/nll.npy")
        print(f"    NLL values: {nll}")
        print(f"    Total NLL: {np.sum(nll)}")
    except Exception as e:
        print(f"    ERROR during NLL calculation: {e}")

    # B. Earth Mover's Distance (EMD / Kantorovich)
    print("\n[2] Calculating Earth Mover's Distance (EMD)...")
    try:
        emds = eval_utils.evaluate_kantorovich(device, args, model, growth_model)
        print(f"    EMDs per timepoint saved to {args.save}/emds.npy")
        print(f"    EMD values: {emds}")
        print(f"    Mean EMD: {np.mean(emds)}")
    except Exception as e:
        print(f"    ERROR during EMD calculation: {e}")

    # C. EMD for Leave-out Timepoint (If applicable)
    if hasattr(args, 'leaveout_timepoint') and args.leaveout_timepoint >= 0:
        print(f"\n[3] Calculating EMD for Leave-out Timepoint {args.leaveout_timepoint} (v2)...")
        try:
            emds_v2 = eval_utils.evaluate_kantorovich_v2(device, args, model, growth_model)
            print(f"    Leave-out EMDs saved to {args.save}/emds_v2.npy")
            print(f"    EMD (Backward, Forward): {emds_v2}")
        except Exception as e:
            print(f"    ERROR during EMD v2 calculation: {e}")
            
    # D. Mean Squared Error (MSE - Only if you have ground truth paths)
    # Check if your data object has a 'get_paths' method and it returns data
    has_paths = hasattr(data, 'get_paths') and data.get_paths() is not None
    if has_paths:
        print("\n[4] Calculating Mean Squared Error (MSE)...")
        try:
            mses = eval_utils.evaluate_mse(device, args, model, growth_model)
            print(f"    MSEs per timepoint saved to {args.save}/mses.npy")
            print(f"    MSE values: {mses}")
            print(f"    Mean MSE: {np.mean(mses)}")
        except Exception as e:
            print(f"    ERROR during MSE calculation: {e}")
    else:
        print("\n[4] Skipping MSE calculation (No ground truth paths found).")


    # E. Generate Samples for Visualization / EMD check
    print("\n[5] Generating Samples for Last Timepoint...")
    try:
        # Evaluate on the last timepoint
        last_tp_index = -1 # Or choose a specific index
        eval_utils.generate_samples(device, args, model, growth_model, timepoint=args.timepoints[last_tp_index])
        print(f"    Samples and plot saved in {args.save}")
    except Exception as e:
        print(f"    ERROR during sample generation: {e}")

    # --- 6. Run Visualizations (Optional, especially if 2D) ---
    if data.get_shape()[0] == 2:
        print("\n[6] Generating 2D Visualizations...")
        try:
            plot_output(device, args, model, data)
        except Exception as e:
            print(f"    ERROR during visualization: {e}")
    else:
        print("\n[6] Skipping 2D visualizations (Data is not 2D).")


    print("\n--- Evaluation Finished ---")


if __name__ == "__main__":
    args = parser.parse_args()
    
    # IMPORTANT: Ensure your --save argument points to the directory
    #            where your 'checkpt.pth' and logs are.
    if not args.save:
        print("ERROR: You must provide the --save argument pointing to your trained model directory.")
    else:
        # You might need to load args from the saved run or ensure they match
        # For now, we assume the provided args match the training run.
        # Ideally, you'd save 'args' during training and load them here.
        main(args)