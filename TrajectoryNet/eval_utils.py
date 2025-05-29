import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import ot as pot
import csv

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
    """generates samples using model and base density

    This is useful for measuring the wasserstein distance between the
    predicted distribution and the true distribution for evaluation
    purposes against other types of models. We should use
    negative log likelihood if possible as it is deterministic and
    more discriminative for this model type.

    TODO: Is this biased???
    """
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
        plt.scatter(z[:, 0], z[:, 1], s=0.1, alpha=0.5)
        original_data = args.data.get_data()[args.data.get_times() == timepoint]
        idx = np.random.randint(original_data.shape[0], size=n)
        samples = original_data[idx, :]
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
        plt.savefig(os.path.join(args.save, "samples%d.png" % timepoint))
        plt.close()

        pz = np.exp(logpz)
        pz = pz / np.sum(pz)
        print(pz)

        print(
            earth_mover_distance(
                original_data, samples + np.random.randn(*samples.shape) * 0.1
            )
        )

        print(earth_mover_distance(z, original_data))
        print(earth_mover_distance(z, samples))
        # print(earth_mover_distance(z, original_data, weights1=pz.flatten()))
        # print(
        #    earth_mover_distance(
        #        args.data.get_data()[args.data.get_times() == (timepoint - 1)],
        #        original_data,
        #    )
        # )

    if args.use_growth and growth_model is not None:
        raise NotImplementedError(
            "generating samples with growth model is not yet implemented"
        )


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

# ======================== Comprehensive Evaluation ========================

def comprehensive_evaluation(device, args, model, growth_model, logger, save_dir):
    """
    Run all evaluation metrics after training
    """
    logger.info("Starting comprehensive evaluation...")
    model.eval()
    eval_results = {}
    
    # 1. Negative Log-Likelihood Evaluation
    logger.info("Computing negative log-likelihood...")
    try:
        nll_losses = evaluate(device, args, model, growth_model)
        eval_results['nll'] = nll_losses
        logger.info(f"NLL losses: {nll_losses}")
    except Exception as e:
        logger.error(f"NLL evaluation failed: {e}")
    
    # 2. Kantorovich Distance Evaluation
    logger.info("Computing Kantorovich distances...")
    try:
        emds = evaluate_kantorovich(device, args, model, growth_model, n=5000)
        eval_results['kantorovich'] = emds
        logger.info(f"Kantorovich distances: {emds}")
    except Exception as e:
        logger.error(f"Kantorovich evaluation failed: {e}")
    
    # 3. Kantorovich Distance V2 (if leaveout timepoint specified)
    if args.leaveout_timepoint >= 0 and args.leaveout_timepoint < len(args.timepoints) - 1:
        logger.info("Computing Kantorovich distances V2...")
        try:
            emds_v2 = evaluate_kantorovich_v2(device, args, model, growth_model)
            eval_results['kantorovich_v2'] = emds_v2
            logger.info(f"Kantorovich V2 distances: {emds_v2}")
        except Exception as e:
            logger.error(f"Kantorovich V2 evaluation failed: {e}")
    
    # 4. MSE Evaluation (if path data available)
    if hasattr(args.data, 'get_paths'):
        logger.info("Computing MSE on paths...")
        try:
            mses = evaluate_mse(device, args, model, growth_model)
            eval_results['mse'] = mses
            logger.info(f"Path MSE: {np.mean(mses)}")
        except Exception as e:
            logger.error(f"MSE evaluation failed: {e}")
    
    # 5. Path Length Analysis
    logger.info("Computing path lengths...")
    try:
        path_lengths = []
        for tp_idx, tp in enumerate(args.timepoints):
            if tp > 0:  # Skip base timepoint
                path_length = calculate_path_length(
                    device, args, model, args.data, args.int_tps[tp_idx]
                )
                path_lengths.append(path_length)
                logger.info(f"Path length to t={tp}: {path_length:.4f}")
        eval_results['path_lengths'] = path_lengths
    except Exception as e:
        logger.error(f"Path length calculation failed: {e}")
    
    # 6. Generate samples for each timepoint
    logger.info("Generating samples for visualization...")
    try:
        for tp_idx, tp in enumerate(args.timepoints):
            generate_samples(device, args, model, growth_model, n=2000, timepoint=tp_idx)
            logger.info(f"Generated samples for t={tp}")
    except Exception as e:
        logger.error(f"Sample generation failed: {e}")
    
    # Save evaluation summary
    eval_summary_path = os.path.join(save_dir, "evaluation_summary.txt")
    with open(eval_summary_path, "w") as f:
        f.write("Comprehensive Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        
        for metric_name, values in eval_results.items():
            f.write(f"{metric_name.upper()}:\n")
            if isinstance(values, (list, np.ndarray)):
                f.write(f"  Values: {values}\n")
                if len(values) > 0:
                    f.write(f"  Mean: {np.mean(values):.6f}\n")
                    f.write(f"  Std: {np.std(values):.6f}\n")
            else:
                f.write(f"  Value: {values}\n")
            f.write("\n")
    
    logger.info(f"Evaluation summary saved to {eval_summary_path}")
    return eval_results

# ======================== Visualization Functions ========================

def plot_vector_fields_with_samples(device, args, model, logger, save_dir):
    """
    Plot vector fields with forward-integrated samples from base distribution
    """
    if args.data.get_shape()[0] > 2:
        logger.warning("Skipping vector field visualization as data dimension > 2")
        return
    
    logger.info("Plotting vector fields with integrated samples...")
    n_samples = 500
    utils.makedirs(os.path.join(save_dir, "vector_fields"))
    
    # Create base samples
    base_samples = args.data.base_sample()(n_samples, *args.data.get_shape())
    base_samples = base_samples.numpy()
    
    with torch.no_grad():
        # For each timepoint, integrate samples forward and plot vector field
        for i, (tp, itp) in enumerate(zip(args.timepoints, args.int_tps)):
            plt.figure(figsize=(12, 8))
            
            # Get data at this timepoint
            data_at_tp = args.data.get_data()[args.data.get_times() == tp]
            
            # Integrate base samples forward to this timepoint
            z_samples = torch.from_numpy(base_samples).type(torch.float32).to(device)
            
            # Forward integration from base to timepoint
            int_list = []
            for j in range(i + 1):
                if j == 0:
                    prev_time = 0.0
                else:
                    prev_time = args.int_tps[j - 1]
                int_times = torch.tensor([prev_time, args.int_tps[j]]).type(torch.float32).to(device)
                int_list.append(int_times)
            
            # Apply forward transformations
            z = z_samples
            for int_times in int_list:
                z = model(z, integration_times=int_times, reverse=True)
            
            integrated_samples = z.cpu().numpy()
            
            # Create grid for vector field
            x_min = min(data_at_tp[:, 0].min(), integrated_samples[:, 0].min()) - 1
            x_max = max(data_at_tp[:, 0].max(), integrated_samples[:, 0].max()) + 1
            y_min = min(data_at_tp[:, 1].min(), integrated_samples[:, 1].min()) - 1
            y_max = max(data_at_tp[:, 1].max(), integrated_samples[:, 1].max()) + 1
            
            x_grid = np.linspace(x_min, x_max, 20)
            y_grid = np.linspace(y_min, y_max, 20)
            X, Y = np.meshgrid(x_grid, y_grid)
            grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
            
            # Compute vector field at grid points
            grid_tensor = torch.from_numpy(grid_points).type(torch.float32).to(device)
            time_tensor = torch.tensor(itp).type(torch.float32).to(device)
            
            try:
                # Get vector field from ODE function
                vectors = -model.chain[0].odefunc.odefunc.diffeq(time_tensor, grid_tensor)
                vectors = vectors.cpu().numpy()
                
                U = vectors[:, 0].reshape(X.shape)
                V = vectors[:, 1].reshape(Y.shape)
                
                # Plot vector field
                plt.quiver(X, Y, U, V, alpha=0.6, color='gray', scale=20, width=0.003)
            except Exception as e:
                logger.warning(f"Could not plot vector field: {e}")
            
            # Plot data points
            plt.scatter(data_at_tp[:, 0], data_at_tp[:, 1], 
                       c='red', s=20, alpha=0.7, label=f'Data t={tp}')
            
            # Plot integrated samples
            plt.scatter(integrated_samples[:, 0], integrated_samples[:, 1], 
                       c='blue', s=15, alpha=0.5, label='Forward samples')
            
            plt.title(f'Vector Field and Samples at t={tp}')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            fig_filename = os.path.join(save_dir, "vector_fields", f"vf_t{tp:02d}.png")
            plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved vector field plot for t={tp}")

def plot_output(device, args, model, save_dir):
    """Plots trajectories and densities for 2D data"""
    if args.data.get_shape()[0] != 2:
        return
        
    save_traj_dir = os.path.join(save_dir, "trajectory")
    utils.makedirs(save_traj_dir)
    
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