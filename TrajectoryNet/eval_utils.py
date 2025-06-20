import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import ot as pot
import csv
from scipy.stats import wasserstein_distance

# Import visualization utilities
from TrajectoryNet.lib.viz_scrna import (
    save_vectors,
    save_trajectory,
    trajectory_to_video,
    save_trajectory_density
)

# Import EMD calculation
from .optimal_transport.emd import earth_mover_distance

matplotlib.use('Agg')

# ======================== Core Evaluation Metrics ========================

def generate_samples(device, args, model, growth_model, n=10000, timepoint=None):

    
    z_samples = args.data.base_sample()(n, *args.data.get_shape()).to(device)
    
    # Forward pass through the model / growth model
    with torch.no_grad():
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in args.int_tps[: timepoint + 1]
        ]

        logpz = args.data.base_density()(z_samples)
        z = z_samples
        for it in int_list:
            z, logpz = model(z, logpz, integration_times=it, reverse=True)
        z = z.cpu().numpy()
        np.save(os.path.join(args.save, "samples_%0.2f.npy" % timepoint), z)
        logpz = logpz.cpu().numpy()
        
        # Create visualization
        plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5, label='Generated samples')
        original_data = args.data.get_data()[args.data.get_times() == timepoint]
        idx = np.random.randint(original_data.shape[0], size=n)
        samples = original_data[idx, :]
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label='Original data')
        plt.legend()
        plt.savefig(os.path.join(args.save, "samples%d.png" % timepoint))
        plt.close()

        # Calculate probability weights for weighted EMD
        pz = np.exp(logpz)
        pz = pz / np.sum(pz)
        print(f"Sample probability statistics - Min: {pz.min():.6f}, Max: {pz.max():.6f}, Mean: {pz.mean():.6f}")

        print("\n" + "="*60)
        print(f"EARTH MOVER DISTANCE EVALUATION (timepoint {timepoint})")
        print("="*60)

        # 1. Noisy baseline EMD
        noisy_samples = samples + np.random.randn(*samples.shape) * 0.1
        emd_noisy = earth_mover_distance(original_data, noisy_samples)
        print(f"1. NOISY BASELINE EMD: {emd_noisy:.6f}")
        print("   → Distance between original data and real samples + Gaussian noise")
        print("   → Shows expected EMD for 'reasonable' but imperfect samples")
        print()

        # 2. Standard unweighted EMD (generated vs original)
        emd_unweighted = earth_mover_distance(z, original_data)
        print(f"2. UNWEIGHTED EMD (Generated vs Original): {emd_unweighted:.6f}")
        print("   → Standard distance between model-generated samples and real data")
        print("   → Lower is better - measures how realistic the generated samples are")
        print()

        # 3. Weighted EMD (using model probabilities)
        emd_weighted = earth_mover_distance(z, original_data, weights1=pz.flatten())
        print(f"3. WEIGHTED EMD (Generated vs Original): {emd_weighted:.6f}")
        print("   → Uses model's probability estimates as sample weights")
        print("   → Emphasizes samples the model considers more likely")
        print("   → More sophisticated measure of model quality")
        print()

        # 4. Generated vs sampled real data
        emd_gen_vs_samples = earth_mover_distance(z, samples)
        print(f"4. GENERATED vs REAL SAMPLES EMD: {emd_gen_vs_samples:.6f}")
        print("   → Distance between generated samples and randomly selected real samples")
        print("   → Direct comparison of model output vs ground truth")
        print()

        # 5. Temporal EMD (between consecutive timepoints)
        if timepoint > 0:
            prev_data = args.data.get_data()[args.data.get_times() == (timepoint - 1)]
            emd_temporal = earth_mover_distance(prev_data, original_data)
            print(f"5. TEMPORAL EMD (t-1 vs t): {emd_temporal:.6f}")
            print(f"   → Natural change in data between timepoint {timepoint-1} and {timepoint}")
            print("   → Baseline for expected temporal variation in the dataset")
            print()
        else:
            print("5. TEMPORAL EMD: N/A (timepoint = 0, no previous timepoint available)")
            print()

        print("="*60)
        print("INTERPRETATION GUIDE:")
        print("• Lower EMD values indicate better similarity between distributions")
        print("• Compare Generated vs Original EMD to Noisy Baseline EMD")
        print("• Weighted EMD accounts for model confidence and may be more reliable")
        print("• Temporal EMD shows natural data variation over time")
        print("="*60)


    if args.use_growth and growth_model is not None:
        raise NotImplementedError(
            "generating samples with growth model is not yet implemented"
        )

def earth_mover_distance(samples1, samples2):
    """Compute Earth Mover's Distance between two sets of 2D samples"""
    emd_x = wasserstein_distance(samples1[:, 0], samples2[:, 0])
    emd_y = wasserstein_distance(samples1[:, 1], samples2[:, 1])
    return (emd_x + emd_y) / 2

def evaluate_visualize(device, args, model, growth_model, n=10000, timepoint=None, grid_size=20):
    """
    Generate samples, compute EMD and MSE, and visualize with vector fields
    
    Args:
        device: torch device
        args: configuration arguments
        model: trained neural ODE model
        growth_model: growth model (unused)
        n: number of samples to generate
        timepoint: evaluation timepoint
        grid_size: resolution for vector field grid
    """
    # Create output directory
    savedir = args.save   
    os.makedirs(savedir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # 1. Generate samples from the model
    z_samples = args.data.base_sample()(n, *args.data.get_shape()).to(device)
    
    with torch.no_grad():
        int_list = [
            torch.tensor([it - args.time_scale, it]).float().to(device)
            for it in args.int_tps[: timepoint + 1]
        ]

        logpz = args.data.base_density()(z_samples)
        z = z_samples
        for it in int_list:
            z, logpz = model(z, logpz, integration_times=it, reverse=True)
        generated = z.cpu().numpy()
    
    # 2. Get real data samples
    original_data = args.data.get_data()[args.data.get_times() == timepoint]
    real_samples = original_data[np.random.choice(original_data.shape[0], n, replace=False)]
    
    # 3. Compute Earth Mover's Distance
    emd = earth_mover_distance(generated, original_data)
    
    # 4. Compute Mean Squared Error
    # For MSE, we need to match the number of samples
    min_samples = min(generated.shape[0], real_samples.shape[0])
    generated_subset = generated[:min_samples]
    real_subset = real_samples[:min_samples]
    mse = np.mean((generated_subset - real_subset) ** 2)
    
    # 6. Save results to CSV file (but don't print yet)
    results_file = os.path.join(savedir, "evaluation_results.csv")
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(results_file)
    
    # Append results to CSV
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header if file doesn't exist
            writer.writerow(['timepoint', 'emd', 'mse'])
        writer.writerow([timepoint, emd, mse])
    
    # 7. Create vector field grid
    K = grid_size * 1j
    y, x = np.mgrid[-4:4:K, -4:4:K]
    grid_points = torch.tensor(np.stack([x.ravel(), y.ravel()], axis=1), 
                              dtype=torch.float32, 
                              device=device)
    logps = torch.zeros(grid_points.shape[0], 1, device=device, dtype=torch.float32)
    
    # 8. Compute vector field at final timepoint
    with torch.no_grad():
        # Create time tensor (critical fix)
        t_tensor = torch.tensor([timepoint], device=device, dtype=torch.float32)
        
        # Get ODE function from model
        if hasattr(model, 'odefunc'):
            odefunc = model.odefunc
        elif hasattr(model, 'chain') and hasattr(model.chain[0], 'odefunc'):
            odefunc = model.chain[-1].odefunc
        else:
            raise AttributeError("Model missing odefunc attribute")
        
        # Compute vector field
        dydt = odefunc(t_tensor, (grid_points, logps))[0]
        dydt = -dydt.cpu().numpy()  # Reverse for forward transformation
    
    # Reshape vector field results
    dydt = dydt.reshape(grid_size, grid_size, 2)
    magnitude = np.hypot(dydt[..., 0], dydt[..., 1])
    
    # 9. Create single combined visualization
    # Create plots subdirectory
    plots_dir = os.path.join(savedir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # Plot generated samples
    plt.scatter(generated[:, 0], generated[:, 1], s=3, alpha=0.6, 
                color='blue', label='Generated', zorder=2)
    
    # Plot real data samples
    plt.scatter(real_samples[:, 0], real_samples[:, 1], s=3, alpha=0.6, 
                color='red', label='Real Data', zorder=2)
    
    # Plot vector field with bigger arrows
    quiver = plt.quiver(
        x, y, dydt[..., 0], dydt[..., 1], magnitude,
        cmap='viridis', scale=15, width=0.006, pivot='mid',
        angles='xy', scale_units='xy', alpha=0.8, zorder=1
    )
    
    # Add colorbar for vector field magnitude
    cbar = plt.colorbar(quiver, shrink=0.8)
    cbar.set_label('Vector Field Magnitude', rotation=270, labelpad=15)
    
    # Formatting
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title(f"Model Evaluation at t={timepoint}\nEMD: {emd:.4f} | MSE: {mse:.4f}", 
              fontsize=14, pad=20)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.legend(markerscale=3, loc='upper right')
    plt.grid(alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save plot in subdirectory and close
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"evaluation_{timepoint:.2f}.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()

#helper function to display csv data 
def display_results_table(results_file):
    """
    Read CSV file and display formatted table of results
    
    Args:
        results_file: path to CSV file containing results
    """
    if not os.path.exists(results_file):
        return
    
    # Read CSV file
    with open(results_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        rows = list(reader)
    
    if not rows:
        return
    
    # Extract data
    timepoints = [float(row[0]) for row in rows]
    emd_values = [float(row[1]) for row in rows]
    mse_values = [float(row[2]) for row in rows]
    
    # Create header
    print("\n" + "="*80)
    print("EVALUATION RESULTS ACROSS TIMEPOINTS")
    print("="*80)
    
    # Print timepoint row
    print(f"{'Timepoint':<12}", end="")
    for tp in timepoints:
        print(f"{tp:<12.2f}", end="")
    print()
    
    # Print separator
    print("-"*80)
    
    # Print EMD row
    print(f"{'EMD':<12}", end="")
    for emd in emd_values:
        print(f"{emd:<12.6f}", end="")
    print()
    
    # Print MSE row
    print(f"{'MSE':<12}", end="")
    for mse in mse_values:
        print(f"{mse:<12.6f}", end="")
    print()
    
    print("="*80)    
    
def calculate_path_length(device, args, model, data, end_time, n_pts=10000):
    """Calculates the total length of the path from time 0 to timepoint"""
    # z_samples = torch.tensor(data.get_data()).type(torch.float32).to(device)
    z_samples = data.base_sample()(n_pts, *data.get_shape()).to(device)
    model.eval()
    n = 1001
    with torch.no_grad():
        integration_times = (
            torch.tensor(np.linspace(0, end_time, n)).type(torch.float32).to(device)
        )
        # z, _ = model(z_samples, torch.zeros_like(z_samples), integration_times=integration_times, reverse=False)
        z, _ = model(
            z_samples,
            torch.zeros_like(z_samples),
            integration_times=integration_times,
            reverse=True,
        )
        z = z.cpu().numpy()
        z_diff = np.diff(z, axis=0)
        z_lengths = np.sum(np.linalg.norm(z_diff, axis=-1), axis=0)
        total_length = np.mean(z_lengths)
        import ot as pot
        from scipy.spatial.distance import cdist

        emd = pot.emd2(
            np.ones(n_pts) / n_pts,
            np.ones(n_pts) / n_pts,
            cdist(z[-1, :, :], data.get_data()),
        )
        print(total_length, emd)
        plt.scatter(z[-1, :, 0], z[-1, :, 1])
        plt.savefig("test.png")
        plt.close()


def evaluate_mse(device, args, model, growth_model=None):
    if args.use_growth or growth_model is not None:
        print("WARNING: Ignoring growth model and computing anyway")

    paths = args.data.get_paths()

    z_samples = torch.tensor(paths[:, 0, :]).type(torch.float32).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in args.int_tps
        ]

        logpz = args.data.base_density()(z_samples)
        z = z_samples
        zs = []
        for it in int_list:
            z, _ = model(z, logpz, integration_times=it, reverse=True)
            zs.append(z.cpu().numpy())
        zs = np.stack(zs)
        np.save(os.path.join(args.save, "path_samples.npy"), zs)

        # logpz = logpz.cpu().numpy()
        # plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5)
        mses = []
        print(zs.shape, paths[:, 1, :].shape)
        for tpi in range(len(args.timepoints)):
            mses.append(np.mean((paths[:, tpi + 1, :] - zs[tpi]) ** 2, axis=(-2, -1)))
        mses = np.array(mses)
        print(mses.shape)
        np.save(os.path.join(args.save, "mses.npy"), mses)
        return mses

# Helper function to load real data for selected timepoint indices from an NPZ file
def _load_real_data_for_selected_indices(dataset,
                                         data_file_indices_to_load,
                                         num_samples_to_use_per_timepoint,
                                         phate_key='phate',
                                         labels_key='sample_labels'):
    """
    Loads data from an NPZ file for specified timepoint indices.
    These indices refer to the sorted unique timepoint labels found in the data file.

    Args:
        dataset (str): Path to the .npz data file.
        data_file_indices_to_load (list or set): A collection of 0-based indices
                                                corresponding to the sorted unique
                                                timepoint labels in the data file.
                                                E.g., if file labels are [t_a, t_b, t_c] (sorted),
                                                index 0 is for t_a, index 1 for t_b, etc.
        num_samples_to_use_per_timepoint (int): Number of samples to retrieve for each timepoint.
        phate_key (str): Key for the data array in the NPZ file.
        labels_key (str): Key for the sample labels array in the NPZ file.

    Returns:
        dict: {data_file_index: data_array_for_that_index}
              Each data_array will have shape (num_samples_to_use_per_timepoint, n_features).

    Raises:
        ValueError: If data is insufficient or indices are out of range.
        FileNotFoundError: If dataset does not exist.
    """
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Data file not found: {dataset}")

    try:
        raw_data_content = np.load(dataset)
    except Exception as e:
        raise IOError(f"Could not read NPZ file at {dataset}: {e}")


    if phate_key not in raw_data_content:
        raise ValueError(f"Data key '{phate_key}' not found in NPZ file: {dataset}")
    if labels_key not in raw_data_content:
        raise ValueError(f"Labels key '{labels_key}' not found in NPZ file: {dataset}")

    all_phate_data = raw_data_content[phate_key]
    all_sample_labels = raw_data_content[labels_key]

    unique_sorted_labels_in_file = sorted(list(np.unique(all_sample_labels)))

    max_data_idx_available = len(unique_sorted_labels_in_file) - 1
    for requested_idx in data_file_indices_to_load:
        if not (0 <= requested_idx <= max_data_idx_available):
            raise ValueError(
                f"Requested data file index {requested_idx} is out of range. "
                f"Data has {len(unique_sorted_labels_in_file)} unique timepoints "
                f"(indices 0 to {max_data_idx_available})."
            )

    loaded_data_slices = {}
    for data_idx in data_file_indices_to_load:
        actual_label_value_to_fetch = unique_sorted_labels_in_file[data_idx]
        data_for_this_label = all_phate_data[all_sample_labels == actual_label_value_to_fetch]

        if data_for_this_label.shape[0] < num_samples_to_use_per_timepoint:
            raise ValueError(
                f"Not enough samples for time label '{actual_label_value_to_fetch}' "
                f"(corresponding to data file index {data_idx}). "
                f"Found {data_for_this_label.shape[0]}, requested {num_samples_to_use_per_timepoint}."
            )

        # Consistently take the first N samples.
        # Consider random sampling if appropriate for your use case.
        loaded_data_slices[data_idx] = data_for_this_label[:num_samples_to_use_per_timepoint, :]

    return loaded_data_slices

def evaluate_mse_timepoint(device, args, model, target_timepoints, growth_model=None):
    """
    Evaluate MSE between model predictions and real data at specific timepoints.
    Computes average MSE over all trajectories for each timepoint.

    MODIFIED: This version loads real data only for the required timepoints directly
              from an .npz file specified in `args`. It no longer uses `args.data.get_paths()`.

    Args:
        device: torch device.
        args: Arguments object/namespace. Expected to contain:
                args.dataset (str): Path to the .npz data file.
                args.int_tps (list): List of time values for integration endpoints.
                                     e.g., [t1, t2, t3]. Model integrates from initial to t1, etc.
                args.time_scale (float): Value used to define integration intervals
                                        [t - time_scale, t] for each t in args.int_tps.
                args.save (str): Directory path to save results.
                args.data.base_density (callable): Function to compute base density for logpz.
                args.phate_key (str, optional): Key for data in NPZ. Defaults to 'phate'.
                args.labels_key (str, optional): Key for labels in NPZ. Defaults to 'sample_labels'.
        model: Trained neural ODE model.
        target_timepoints (list of int): List of timepoint INDICES to evaluate.
                                         These indices correspond to `args.int_tps`.
                                         E.g., if target_timepoints=[0, 2], evaluation is for
                                         model predictions at times `args.int_tps[0]` and `args.int_tps[2]`.
        growth_model: Growth model (ignored with warning if provided).

    Returns:
        dict: {target_timepoint_index: average_mse_value}
              e.g., {0: mse_for_args.int_tps[0], 2: mse_for_args.int_tps[2]}
    """
    if hasattr(args, 'use_growth') and args.use_growth or growth_model is not None:
        print("INFO: Ignoring growth model parameter as it's not used in this MSE evaluation.")

    if not target_timepoints:
        print("INFO: No target timepoints specified. Returning empty results.")
        return {}

    # Validate target_timepoints (these are indices for args.int_tps)
    max_allowable_target_idx = len(args.int_tps) - 1
    for tp_idx in target_timepoints:
        if not (0 <= tp_idx <= max_allowable_target_idx):
            raise ValueError(
                f"Target timepoint index {tp_idx} is out of range. "
                f"With args.int_tps of length {len(args.int_tps)}, "
                f"valid indices are [0, {max_allowable_target_idx}]."
            )

    # Determine which data file indices are needed for real data comparison.
    # Based on original logic:
    # - Initial conditions correspond to data file index 0.
    # - Real data for args.int_tps[tp_idx] corresponds to data file index (tp_idx + 1).
    required_data_file_indices = {0}  # For initial conditions
    for tp_idx in target_timepoints:
        required_data_file_indices.add(tp_idx + 1)

    # --- Data Loading Modification ---
    if not hasattr(args, 'dataset'):
        raise AttributeError(
            "For modified data loading, 'args' must contain 'dataset' (str) "
        )

    phate_key = getattr(args, 'phate_key', 'phate')
    labels_key = getattr(args, 'labels_key', 'sample_labels')
    num_samples_per_tp = 100

    loaded_real_data_slices = _load_real_data_for_selected_indices(
        args.dataset,
        sorted(list(required_data_file_indices)), # Ensure uniqueness and order
        num_samples_per_tp,
        phate_key=phate_key,
        labels_key=labels_key
    )
    # `loaded_real_data_slices` is a dict: {data_file_idx: data_array}
    # --- End Data Loading Modification ---

    initial_conditions_from_file = loaded_real_data_slices[0]
    n_trajectories = initial_conditions_from_file.shape[0]

    print(f"Evaluating MSE for {n_trajectories} trajectories (samples).")
    print(f"Target timepoint indices (referring to args.int_tps): {target_timepoints}")

    z_samples_tensor = torch.tensor(initial_conditions_from_file, dtype=torch.float32).to(device)

    # Forward pass through the model to get predictions
    with torch.no_grad():
        integration_intervals = []
        for t_end in args.int_tps:
            # This interval definition is from your original code.
            # It implies args.time_scale defines the duration or start relative to t_end.
            t_start = t_end - args.time_scale
            integration_intervals.append(torch.tensor([t_start, t_end], dtype=torch.float32).to(device))

        if not hasattr(args, 'data') or not hasattr(args.data, 'base_density'):
             raise AttributeError(
                 "args.data.base_density() callable is required for logpz calculation. "
                 "Ensure it's correctly passed in `args`."
            )

        current_logpz = args.data.base_density()(z_samples_tensor)
        current_z = z_samples_tensor

        model_predictions_over_time = [current_z.cpu().numpy()] # Store initial conditions

        for interval in integration_intervals:
            # Assuming 'reverse=True' is part of your model's generation process as in original.
            current_z, current_logpz = model(current_z, current_logpz, integration_times=interval, reverse=True)
            model_predictions_over_time.append(current_z.cpu().numpy())

        # predictions_at_each_step[0] is the initial state.
        # predictions_at_each_step[k] is state after k-th integration, corresponds to time args.int_tps[k-1].
        predictions_at_each_step = np.stack(model_predictions_over_time)

    # Compute MSE for each specified target timepoint index
    mse_results = {}

    print("\n" + "="*50)
    print("MSE EVALUATION AT SPECIFIC TIMEPOINTS")
    print("="*50 + "\n")

    for tp_idx in target_timepoints: # tp_idx is an index for `args.int_tps`
        # Predicted data at the time `args.int_tps[tp_idx]`
        # This corresponds to predictions_at_each_step[tp_idx + 1]
        predicted_data_at_target = predictions_at_each_step[tp_idx + 1]

        # Real data for the time `args.int_tps[tp_idx]`
        # This corresponds to data file index `tp_idx + 1`
        real_data_file_index_for_comparison = tp_idx + 1
        real_data_at_target = loaded_real_data_slices[real_data_file_index_for_comparison]

        mean_squared_error = np.mean((predicted_data_at_target - real_data_at_target) ** 2)
        mse_results[tp_idx] = mean_squared_error

        print(f"Timepoint Index {tp_idx} (corresponds to actual time {args.int_tps[tp_idx]:.2f}): "
              f"Average MSE = {mean_squared_error:.6f}")

    print("\n" + "="*50)
    print("• Lower MSE indicates better prediction accuracy.")
    print("• Compare MSE across timepoints to understand temporal performance.")
    print("="*50 + "\n")

    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
        print(f"Created save directory: {args.save}")

    results_save_path = os.path.join(args.save, "mse_at_selected_timepoints.npy")
    try:
        np.save(results_save_path, mse_results)
        print(f"MSE results dictionary saved to: {results_save_path}")
    except Exception as e:
        print(f"Error saving results to {results_save_path}: {e}")


    return mse_results

def evaluate_kantorovich_v2(device, args, model, growth_model=None):
    """Eval the model via kantorovich distance on leftout timepoint

    v2 computes samples from subsequent timepoint instead of base distribution.
    this is arguably a fairer comparison to other methods such as WOT which are
    not model based this should accumulate much less numerical error in the
    integration procedure. However fixes to the number of samples to the number in the
    previous timepoint.

    If we have a growth model we should use this to modify the weighting of the
    points over time.
    """
    if args.use_growth or growth_model is not None:
        # raise NotImplementedError(
        #    "generating samples with growth model is not yet implemented"
        # )
        print("WARNING: Ignoring growth model and computing anyway")

    # Backward pass through the model / growth model
    with torch.no_grad():
        int_times = torch.tensor(
            [
                args.int_tps[args.leaveout_timepoint],
                args.int_tps[args.leaveout_timepoint + 1],
            ]
        )
        int_times = int_times.type(torch.float32).to(device)
        next_z = args.data.get_data()[
            args.data.get_times() == args.leaveout_timepoint + 1
        ]
        next_z = torch.from_numpy(next_z).type(torch.float32).to(device)
        prev_z = args.data.get_data()[
            args.data.get_times() == args.leaveout_timepoint - 1
        ]
        prev_z = torch.from_numpy(prev_z).type(torch.float32).to(device)
        zero = torch.zeros(next_z.shape[0], 1).to(device)
        z_backward, _ = model.chain[0](next_z, zero, integration_times=int_times)
        z_backward = z_backward.cpu().numpy()
        int_times = torch.tensor(
            [
                args.int_tps[args.leaveout_timepoint - 1],
                args.int_tps[args.leaveout_timepoint],
            ]
        )
        zero = torch.zeros(prev_z.shape[0], 1).to(device)
        z_forward, _ = model.chain[0](
            prev_z, zero, integration_times=int_times, reverse=True
        )
        z_forward = z_forward.cpu().numpy()

        emds = []
        for tpi in [args.leaveout_timepoint]:
            original_data = args.data.get_data()[
                args.data.get_times() == args.timepoints[tpi]
            ]
            emds.append(earth_mover_distance(z_backward, original_data))
            emds.append(earth_mover_distance(z_forward, original_data))

        emds = np.array(emds)
        np.save(os.path.join(args.save, "emds_v2.npy"), emds)
        return emds


def evaluate_kantorovich(device, args, model, growth_model=None, n=10000):
    """Eval the model via kantorovich distance on all timepoints

    compute samples forward from the starting parametric distribution keeping track
    of growth rate to scale the final distribution.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.

    If we have a growth model we should use this to modify the weighting of the
    points over time.
    """
    if args.use_growth or growth_model is not None:
        # raise NotImplementedError(
        #    "generating samples with growth model is not yet implemented"
        # )
        print("WARNING: Ignoring growth model and computing anyway")

    z_samples = args.data.base_sample()(n, *args.data.get_shape()).to(device)
    # Forward pass through the model / growth model
    with torch.no_grad():
        int_list = []
        for i, it in enumerate(args.int_tps):
            if i == 0:
                prev = 0.0
            else:
                prev = args.int_tps[i - 1]
            int_list.append(torch.tensor([prev, it]).type(torch.float32).to(device))

        # int_list = [
        #    torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
        #    for it in args.int_tps
        # ]
        print(args.int_tps)

        logpz = args.data.base_density()(z_samples)
        z = z_samples
        zs = []
        growthrates = [torch.ones(z_samples.shape[0], 1).to(device)]
        for it, tp in zip(int_list, args.timepoints):
            z, _ = model(z, logpz, integration_times=it, reverse=True)
            zs.append(z.cpu().numpy())
            if args.use_growth:
                time_state = tp * torch.ones(z.shape[0], 1).to(device)
                full_state = torch.cat([z, time_state], 1)
                # Multiply growth rates together to get total mass along path
                growthrates.append(
                    torch.clamp(growth_model(full_state), 1e-4, 1e4) * growthrates[-1]
                )
        zs = np.stack(zs)
        if args.use_growth:
            growthrates = growthrates[1:]
            growthrates = torch.stack(growthrates)
            growthrates = growthrates.cpu().numpy()
            np.save(os.path.join(args.save, "sample_weights.npy"), growthrates)
        np.save(os.path.join(args.save, "samples.npy"), zs)

        # logpz = logpz.cpu().numpy()
        # plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5)
        emds = []
        for tpi in range(len(args.timepoints)):
            original_data = args.data.get_data()[
                args.data.get_times() == args.timepoints[tpi]
            ]
            if args.use_growth:
                emds.append(
                    earth_mover_distance(
                        zs[tpi], original_data, weights1=growthrates[tpi].flatten()
                    )
                )
            else:
                emds.append(earth_mover_distance(zs[tpi], original_data))

        # Add validation point kantorovich distance evaluation
        if args.data.has_validation_samples():
            for tpi in np.unique(args.data.val_labels):
                original_data = args.data.val_data[
                    args.data.val_labels == args.timepoints[tpi]
                ]
                if args.use_growth:
                    emds.append(
                        earth_mover_distance(
                            zs[tpi], original_data, weights1=growthrates[tpi].flatten()
                        )
                    )
                else:
                    emds.append(earth_mover_distance(zs[tpi], original_data))

        emds = np.array(emds)
        print(emds)
        np.save(os.path.join(args.save, "emds.npy"), emds)
        return emds


def evaluate(device, args, model, growth_model=None):
    """Eval the model via negative log likelihood on all timepoints

    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.
    """
    use_growth = args.use_growth and growth_model is not None

    # Backward pass accumulating losses, previous state and deltas
    deltas = []
    zs = []
    z = None
    for i, (itp, tp) in enumerate(zip(args.int_tps[::-1], args.timepoints[::-1])):
        # tp counts down from last
        integration_times = torch.tensor([itp - args.time_scale, itp])
        integration_times = integration_times.type(torch.float32).to(device)

        x = args.data.get_data()[args.data.get_times() == tp]
        x = torch.from_numpy(x).type(torch.float32).to(device)

        if i > 0:
            x = torch.cat((z, x))
            zs.append(z)
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to previous timepoint
        z, delta_logp = model(x, zero, integration_times=integration_times)
        deltas.append(delta_logp)

    logpz = args.data.base_density()(z)

    # build growth rates
    if use_growth:
        growthrates = [torch.ones_like(logpz)]
        for z_state, tp in zip(zs[::-1], args.timepoints[::-1][1:]):
            # Full state includes time parameter to growth_model
            time_state = tp * torch.ones(z_state.shape[0], 1).to(z_state)
            full_state = torch.cat([z_state, time_state], 1)
            growthrates.append(growth_model(full_state))

    # Accumulate losses
    losses = []
    logps = [logpz]
    for i, (delta_logp, tp) in enumerate(zip(deltas[::-1], args.timepoints)):
        n_cells_in_tp = np.sum(args.data.get_times() == tp)
        logpx = logps[-1] - delta_logp
        if use_growth:
            logpx += torch.log(growthrates[i])
        logps.append(logpx[:-n_cells_in_tp])
        losses.append(-torch.sum(logpx[-n_cells_in_tp:]))
    losses = torch.stack(losses).cpu().numpy()
    np.save(os.path.join(args.save, "nll.npy"), losses)
    return losses


# ======================== Visualization Functions ========================

def plot_output(device, args, model, logger, save_dir):
    """Plots trajectories and densities for 2D data"""
    if args.data.get_shape()[0] != 2:
        return
        
    save_traj_dir = os.path.join(save_dir, "trajectory")
    os.makedirs(save_traj_dir, exist_ok=True)
    
    
    logger.info('Plotting trajectory to {}'.format(save_traj_dir))
    data_samples = args.data.get_data()[args.data.sample_index(2000, 0)]
    start_points = args.data.base_sample()(1000, 2)
    
    save_vectors(
        args.data.base_density(),
        model,
        start_points,
        args.data.get_data(),
        args.data.get_times(),
        save_dir,
        skip_first=(not args.data.known_base_density()),
        device=device,
        end_times=args.int_tps,
        ntimes=100,
    )
    
    save_trajectory(
        args.data.base_density(),
        args.data.base_sample(),
        model,
        data_samples,
        save_traj_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=25,
    )
    
    density_dir = os.path.join(save_dir, "density")
    save_trajectory_density(
        args.data.base_density(),
        model,
        data_samples,
        density_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=25,
        memory=0.1,
    )
    
    if args.save_movie:
        trajectory_to_video(save_traj_dir)
        trajectory_to_video(density_dir)

def plot_loss_curves(save_dir):
    """Plots training and evaluation loss curves"""
    train_loss_path = os.path.join(save_dir, "losses.csv")
    eval_loss_path = os.path.join(save_dir, "train_eval.csv")
    
    if os.path.exists(train_loss_path):
        train_data = np.genfromtxt(train_loss_path, delimiter=',', names=True)
        plt.figure(figsize=(10, 6))
        plt.plot(train_data['Iteration'], train_data['Loss'], label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "training_loss.png"))
        plt.close()
    
    if os.path.exists(eval_loss_path):
        eval_data = np.genfromtxt(eval_loss_path, delimiter=',', names=True)
        plt.figure(figsize=(10, 6))
        plt.plot(eval_data['Iteration'], eval_data['Test_Loss'], label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "validation_loss.png"))
        plt.close()

# ======================== Utility Functions ========================

def save_evaluation_results(results, save_dir):
    """Saves evaluation results to CSV files"""
    # Save NLL results
    if 'nll' in results:
        np.savetxt(os.path.join(save_dir, "nll.csv"), results['nll'], delimiter=',')
    
    # Save Kantorovich distances
    if 'kantorovich' in results:
        np.savetxt(os.path.join(save_dir, "kantorovich.csv"), results['kantorovich'], delimiter=',')
    
    # Save path lengths
    if 'path_lengths' in results:
        np.savetxt(os.path.join(save_dir, "path_lengths.csv"), results['path_lengths'], delimiter=',')

def log_evaluation_summary(results, logger):
    """Logs evaluation summary to console and file"""
    summary = "\n===== Evaluation Summary =====\n"
    for metric, values in results.items():
        if isinstance(values, (list, np.ndarray)):
            summary += f"{metric.upper()}: Mean = {np.mean(values):.4f}, Std = {np.std(values):.4f}\n"
        else:
            summary += f"{metric.upper()}: {values}\n"
    
    logger.info(summary)