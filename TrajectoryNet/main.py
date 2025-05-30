""" main.py

Learns ODE from scrna data with comprehensive evaluation

"""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from TrajectoryNet.lib.growth_net import GrowthNet
from TrajectoryNet.lib import utils
from TrajectoryNet.lib.visualize_flow import visualize_transform
from TrajectoryNet.lib.viz_scrna import (
    save_trajectory,
    trajectory_to_video,
    save_vectors,
)
from TrajectoryNet.lib.viz_scrna import save_trajectory_density

# Import evaluation utilities
from TrajectoryNet.eval_utils import (
    generate_samples,
    calculate_path_length,
    evaluate_mse,
    evaluate_mse_at_timepoints,
    evaluate_kantorovich_v2,
    evaluate_kantorovich,
    evaluate,
    plot_vector_fields_with_samples,
    plot_output,
    plot_loss_curves,
    save_evaluation_results,
    log_evaluation_summary,
)

# from train_misc import standard_normal_logprob
from TrajectoryNet.train_misc import (
    set_cnf_options,
    count_nfe,
    count_parameters,
    count_total_time,
    add_spectral_norm,
    spectral_norm_power_iteration,
    create_regularization_fns,
    get_regularization,
    append_regularization_to_log,
    build_model_tabular,
)

from TrajectoryNet import dataset
from TrajectoryNet.parse import parser

matplotlib.use("Agg")


def get_transforms(device, args, model, integration_times):
    """
    Given a list of integration points,
    returns a function giving integration times
    """

    def sample_fn(z, logpz=None):
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in integration_times
        ]
        if logpz is not None:
            # TODO this works right?
            for it in int_list:
                z, logpz = model(z, logpz, integration_times=it, reverse=True)
            return z, logpz
        else:
            for it in int_list:
                z = model(z, integration_times=it, reverse=True)
            return z

    def density_fn(x, logpx=None):
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in integration_times[::-1]
        ]
        if logpx is not None:
            for it in int_list:
                x, logpx = model(x, logpx, integration_times=it, reverse=False)
            return x, logpx
        else:
            for it in int_list:
                x = model(x, integration_times=it, reverse=False)
            return x

    return sample_fn, density_fn


def compute_loss(device, args, model, growth_model, logger, full_data):
    """
    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.
    """

    # Backward pass accumulating losses, previous state and deltas
    deltas = []
    zs = []
    z = None
    interp_loss = 0.0
    for i, (itp, tp) in enumerate(zip(args.int_tps[::-1], args.timepoints[::-1])):
        # tp counts down from last
        integration_times = torch.tensor([itp - args.time_scale, itp])
        integration_times = integration_times.type(torch.float32).to(device)
        # integration_times.requires_grad = True

        # load data and add noise
        idx = args.data.sample_index(args.batch_size, tp)
        x = args.data.get_data()[idx]
        if args.training_noise > 0.0:
            x += np.random.randn(*x.shape) * args.training_noise
        x = torch.from_numpy(x).type(torch.float32).to(device)

        if i > 0:
            x = torch.cat((z, x))
            zs.append(z)
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to previous timepoint
        z, delta_logp = model(x, zero, integration_times=integration_times)
        deltas.append(delta_logp)

        # Straightline regularization
        # Integrate to random point at time t and assert close to (1 - t) * end + t * start
        if args.interp_reg:
            t = np.random.rand()
            int_t = torch.tensor([itp - t * args.time_scale, itp])
            int_t = int_t.type(torch.float32).to(device)
            int_x = model(x, integration_times=int_t)
            int_x = int_x.detach()
            actual_int_x = x * (1 - t) + z * t
            interp_loss += F.mse_loss(int_x, actual_int_x)
    if args.interp_reg:
        print("interp_loss", interp_loss)

    logpz = args.data.base_density()(z)

    # build growth rates
    if args.use_growth:
        growthrates = [torch.ones_like(logpz)]
        for z_state, tp in zip(zs[::-1], args.timepoints[:-1]):
            # Full state includes time parameter to growth_model
            time_state = tp * torch.ones(z_state.shape[0], 1).to(z_state)
            full_state = torch.cat([z_state, time_state], 1)
            growthrates.append(growth_model(full_state))

    # Accumulate losses
    losses = []
    logps = [logpz]
    for i, delta_logp in enumerate(deltas[::-1]):
        logpx = logps[-1] - delta_logp
        if args.use_growth:
            logpx += torch.log(torch.clamp(growthrates[i], 1e-4, 1e4))
        logps.append(logpx[: -args.batch_size])
        losses.append(-torch.mean(logpx[-args.batch_size :]))
    losses = torch.stack(losses)
    weights = torch.ones_like(losses).to(logpx)
    if args.leaveout_timepoint >= 0:
        weights[args.leaveout_timepoint] = 0
    losses = torch.mean(losses * weights)

    # Direction regularization
    if args.vecint:
        similarity_loss = 0
        for i, (itp, tp) in enumerate(zip(args.int_tps, args.timepoints)):
            itp = torch.tensor(itp).type(torch.float32).to(device)
            idx = args.data.sample_index(args.batch_size, tp)
            x = args.data.get_data()[idx]
            v = args.data.get_velocity()[idx]
            x = torch.from_numpy(x).type(torch.float32).to(device)
            v = torch.from_numpy(v).type(torch.float32).to(device)
            x += torch.randn_like(x) * 0.1
            # Only penalizes at the time / place of visible samples
            direction = -model.chain[0].odefunc.odefunc.diffeq(itp, x)
            if args.use_magnitude:
                similarity_loss += torch.mean(F.mse_loss(direction, v))
            else:
                similarity_loss -= torch.mean(F.cosine_similarity(direction, v))
        logger.info(similarity_loss)
        losses += similarity_loss * args.vecint

    # Density regularization
    if args.top_k_reg > 0:
        density_loss = 0
        tp_z_map = dict(zip(args.timepoints[:-1], zs[::-1]))
        if args.leaveout_timepoint not in tp_z_map:
            idx = args.data.sample_index(args.batch_size, tp)
            x = args.data.get_data()[idx]
            if args.training_noise > 0.0:
                x += np.random.randn(*x.shape) * args.training_noise
            x = torch.from_numpy(x).type(torch.float32).to(device)
            t = np.random.rand()
            int_t = torch.tensor([itp - t * args.time_scale, itp])
            int_t = int_t.type(torch.float32).to(device)
            int_x = model(x, integration_times=int_t)
            samples_05 = int_x
        else:
            # If we are leaving out a timepoint the regularize there
            samples_05 = tp_z_map[args.leaveout_timepoint]

        # Calculate distance to 5 closest neighbors
        # WARNING: This currently fails in the backward pass with cuda on pytorch < 1.4.0
        #          works on CPU. Fixed in pytorch 1.5.0
        # RuntimeError: CUDA error: invalid configuration argument
        # The workaround is to run on cpu on pytorch <= 1.4.0 or upgrade
        cdist = torch.cdist(samples_05, full_data)
        values, _ = torch.topk(cdist, 5, dim=1, largest=False, sorted=False)
        # Hinge loss
        hinge_value = 0.1
        values -= hinge_value
        values[values < 0] = 0
        density_loss = torch.mean(values)
        print("Density Loss", density_loss.item())
        losses += density_loss * args.top_k_reg
    losses += interp_loss
    return losses


def train(
    device, args, model, growth_model, regularization_coeffs, regularization_fns, logger
):
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    full_data = (
        torch.from_numpy(
            args.data.get_data()[args.data.get_times() != args.leaveout_timepoint]
        )
        .type(torch.float32)
        .to(device)
    )

    best_loss = float("inf")
    if args.use_growth:
        growth_model.eval()
    end = time.time()
    for itr in range(1, args.niters + 1):
        model.train()
        optimizer.zero_grad()

        # Train
        if args.spectral_norm:
            spectral_norm_power_iteration(model, 1)

        loss = compute_loss(device, args, model, growth_model, logger, full_data)
        loss_meter.update(loss.item())

        if len(regularization_coeffs) > 0:
            # Only regularize on the last timepoint
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff
                for reg_state, coeff in zip(reg_states, regularization_coeffs)
                if coeff != 0
            )
            loss = loss + reg_loss
        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)

        loss.backward()
        optimizer.step()

        # Eval
        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)
        time_meter.update(time.time() - end)
        tt_meter.update(total_time)

        log_message = (
            "Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) |"
            " NFE Forward {:.0f}({:.1f})"
            " | NFE Backward {:.0f}({:.1f})".format(
                itr,
                time_meter.val,
                time_meter.avg,
                loss_meter.val,
                loss_meter.avg,
                nfef_meter.val,
                nfef_meter.avg,
                nfeb_meter.val,
                nfeb_meter.avg,
            )
        )
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(
                log_message, regularization_fns, reg_states
            )
        logger.info(log_message)

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                train_eval(
                    device, args, model, growth_model, itr, best_loss, logger, full_data
                )

        if itr % args.viz_freq == 0:
            if args.data.get_shape()[0] > 2:
                logger.warning("Skipping vis as data dimension is >2")
            else:
                with torch.no_grad():
                    visualize(device, args, model, itr)
        if itr % args.save_freq == 0:
            chkpt = {
                "state_dict": model.state_dict(),
            }
            if args.use_growth:
                chkpt.update({"growth_state_dict": growth_model.state_dict()})
            utils.save_checkpoint(
                chkpt,
                args.save,
                epoch=itr,
            )
        end = time.time()
    logger.info("Training has finished.")


def train_eval(device, args, model, growth_model, itr, best_loss, logger, full_data):
    model.eval()
    test_loss = compute_loss(device, args, model, growth_model, logger, full_data)
    test_nfe = count_nfe(model)
    log_message = "[TEST] Iter {:04d} | Test Loss {:.6f} |" " NFE {:.0f}".format(
        itr, test_loss, test_nfe
    )
    logger.info(log_message)
    utils.makedirs(args.save)
    with open(os.path.join(args.save, "train_eval.csv"), "a") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow((itr, test_loss))

    if test_loss.item() < best_loss:
        best_loss = test_loss.item()
        chkpt = {
            "state_dict": model.state_dict(),
        }
        if args.use_growth:
            chkpt.update({"growth_state_dict": growth_model.state_dict()})
        torch.save(
            chkpt,
            os.path.join(args.save, "checkpt.pth"),
        )


def visualize(device, args, model, itr):
    model.eval()
    for i, tp in enumerate(args.timepoints):
        idx = args.data.sample_index(args.viz_batch_size, tp)
        p_samples = args.data.get_data()[idx]
        sample_fn, density_fn = get_transforms(
            device, args, model, args.int_tps[: i + 1]
        )
        plt.figure(figsize=(9, 3))
        visualize_transform(
            p_samples,
            args.data.base_sample(),
            args.data.base_density(),
            transform=sample_fn,
            inverse_transform=density_fn,
            samples=True,
            npts=100,
            device=device,
        )
        fig_filename = os.path.join(
            args.save, "figs", "{:04d}_{:01d}.jpg".format(itr, i)
        )
        utils.makedirs(os.path.dirname(fig_filename))
        plt.savefig(fig_filename)
        plt.close()


def run_evaluation(device, args, model, growth_model, logger):
    """
    Run comprehensive evaluation based on command line arguments
    """
    logger.info("Starting evaluation phase...")
    eval_results = {}
    
    # Run specific evaluations based on args
    if getattr(args, 'eval_nll', False):
        logger.info("Computing negative log-likelihood...")
        try:
            nll_losses = evaluate(device, args, model, growth_model)
            eval_results['nll'] = nll_losses
            logger.info(f"NLL losses computed successfully")
        except Exception as e:
            logger.error(f"NLL evaluation failed: {e}")
    
    if getattr(args, 'eval_kantorovich', False):
        logger.info("Computing Kantorovich distances...")
        try:
            emds = evaluate_kantorovich(device, args, model, growth_model, n=5000)
            eval_results['kantorovich'] = emds
            logger.info(f"Kantorovich distances computed successfully")
        except Exception as e:
            logger.error(f"Kantorovich evaluation failed: {e}")
    
    if getattr(args, 'eval_kantorovich_v2', False) and args.leaveout_timepoint >= 0:
        logger.info("Computing Kantorovich distances V2...")
        try:
            emds_v2 = evaluate_kantorovich_v2(device, args, model, growth_model)
            eval_results['kantorovich_v2'] = emds_v2
            logger.info(f"Kantorovich V2 distances computed successfully")
        except Exception as e:
            logger.error(f"Kantorovich V2 evaluation failed: {e}")
    
    if getattr(args, 'eval_mse', False) and hasattr(args.data, 'get_paths'):
        logger.info("Computing MSE on paths...")
        try:
            mses = evaluate_mse(device, args, model, growth_model)
            eval_results['mse'] = mses
            logger.info(f"Path MSE computed successfully")
        except Exception as e:
            logger.error(f"MSE evaluation failed: {e}")
    
    if getattr(args, 'eval_mse_timepoint', False):
        if args.leaveout_timepoint < 0:
            logger.warning(
                f"Skipping MSE at timepoint: 'leaveout_timepoint' ({args.leaveout_timepoint}) is invalid."
            )
        # Check for new dependencies required by the modified function
        elif not (hasattr(args, 'dataset') and \
                  hasattr(args, 'int_tps')): # args.int_tps is used by the function and for logging
            logger.error(
                "Cannot compute MSE at timepoint: 'dataset', "
                "or 'int_tps' missing in args. These are required for the selective loading method."
            )
        else:
            # Ensure leaveout_timepoint is a valid index for args.int_tps before using it for logging
            actual_time_str = "N/A (index out of bounds)"
            is_valid_index_for_logging = 0 <= args.leaveout_timepoint < len(args.int_tps)
            if is_valid_index_for_logging:
                actual_time_str = f"{args.int_tps[args.leaveout_timepoint]:.2f}"

            logger.info(
                f"Computing MSE at target timepoint index {args.leaveout_timepoint} "
                f"(corresponds to time: {actual_time_str}) using selective data loading..."
            )
            try:
                # Call the (modified) evaluate_mse_at_timepoints function
                # It's assumed that the function definition for evaluate_mse_at_timepoints
                # has been updated to the new one that performs selective loading.
                mse_result_dict = evaluate_mse_at_timepoints( # Or evaluate_mse_at_timepoints_modified if you kept a distinct name
                    device=device,
                    args=args,
                    model=model,
                    target_timepoints=[args.leaveout_timepoint], # Pass as a list of indices
                    growth_model=growth_model
                )
                eval_results['mse_timepoint'] = mse_result_dict

                if args.leaveout_timepoint in mse_result_dict:
                    logger.info(
                        f"MSE at timepoint index {args.leaveout_timepoint} "
                        f"(time: {actual_time_str}): " # Use pre-calculated actual_time_str
                        f"{mse_result_dict[args.leaveout_timepoint]:.6f}"
                    )
                else:
                    # This should ideally not happen if target_timepoints was not empty
                    # and the function executed correctly.
                    logger.error(
                        f"Internal error: MSE for timepoint index {args.leaveout_timepoint} "
                        f"not found in results dictionary. Result: {mse_result_dict}"
                    )
            
            except FileNotFoundError as e:
                logger.error(f"MSE timepoint evaluation failed: Data file not found. {e}")
            except ValueError as e: 
                logger.error(f"MSE timepoint evaluation failed: Data loading or value error. {e}")
            except AttributeError as e: 
                logger.error(f"MSE timepoint evaluation failed: Missing or incorrect attribute in args. {e}")
            except IndexError as e: # Catch potential IndexError if leaveout_timepoint is bad for args.int_tps
                logger.error(f"MSE timepoint evaluation failed: 'leaveout_timepoint' ({args.leaveout_timepoint}) "
                             f"is likely an invalid index for 'args.int_tps' (length {len(args.int_tps)}). Error: {e}")
            except Exception as e:
                logger.error(f"MSE timepoint evaluation failed with an unexpected error: {e}", exc_info=True)
    
    
    if getattr(args, 'eval_path_length', False):
        logger.info("Computing path lengths...")
        try:
            for tp_idx, tp in enumerate(args.timepoints):
                if tp > 0:  # Skip base timepoint
                    path_length = calculate_path_length(device, args, model, args.data, args.int_tps[tp_idx])
                    logger.info(f"Path length to t={tp}: computed")
        except Exception as e:
            logger.error(f"Path length calculation failed: {e}")
    
    if getattr(args, 'generate_eval_samples', False):
        logger.info("Generating samples for evaluation...")
        try:
            for tp_idx, tp in enumerate(args.timepoints):
                generate_samples(device, args, model, growth_model, n=2000, timepoint=tp_idx)
                logger.info(f"Generated samples for t={tp}")
        except Exception as e:
            logger.error(f"Sample generation failed: {e}")
    
    # Save and log results if any evaluations were run
    if eval_results:
        save_evaluation_results(eval_results, args.save)
        log_evaluation_summary(eval_results, logger)
    
    return eval_results


def run_visualization(device, args, model, logger):
    """
    Run visualizations based on command line arguments
    """
    logger.info("Starting visualization phase...")
    
    if getattr(args, 'plot_vector_fields', False):
        logger.info("Plotting vector fields with samples...")
        try:
            plot_vector_fields_with_samples(device, args, model, logger, args.save)
            logger.info("Vector field plots generated successfully")
        except Exception as e:
            logger.error(f"Vector field plotting failed: {e}")
    
    if getattr(args, 'plot_trajectories', False):
        logger.info("Plotting trajectory outputs...")
        try:
            if args.data.data.shape[1] == 2:
                plot_output(device, args, model, args.save)
                logger.info("Trajectory plots generated successfully")
            else:
                logger.warning("Skipping trajectory plots - data dimension > 2")
        except Exception as e:
            logger.error(f"Trajectory plotting failed: {e}")
    
    if getattr(args, 'plot_loss', False):
        logger.info("Plotting loss curves...")
        try:
            plot_loss_curves(args.save)
            logger.info("Loss curves plotted successfully")
        except Exception as e:
            logger.error(f"Loss curve plotting failed: {e}")


def main(args):
    # logger
    print(args.no_display_loss)
    utils.makedirs(args.save)
    logger = utils.get_logger(
        logpath=os.path.join(args.save, "logs"),
        filepath=os.path.abspath(__file__),
        displaying=~args.no_display_loss,
    )

    if args.layer_type == "blend":
        logger.info("!! Setting time_scale from None to 1.0 for Blend layers.")
        args.time_scale = 1.0

    logger.info(args)

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if args.use_cpu:
        device = torch.device("cpu")

    args.data = dataset.SCData.factory(args.dataset, args)

    args.timepoints = args.data.get_unique_times()
    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, args.data.get_shape()[0], regularization_fns).to(
        device
    )
    growth_model = None
    if args.use_growth:
        if args.leaveout_timepoint == -1:
            growth_model_path = "../data/externel/growth_model_v2.ckpt"
        elif args.leaveout_timepoint in [1, 2, 3]:
            assert args.max_dim == 5
            growth_model_path = "../data/growth/model_%d" % args.leaveout_timepoint
        else:
            print("WARNING: Cannot use growth with this timepoint")

        growth_model = torch.load(growth_model_path, map_location=device)
    if args.spectral_norm:
        add_spectral_norm(model)
    set_cnf_options(args, model)

    if args.test:
        state_dict = torch.load(args.save + "/checkpt.pth", map_location=device)
        model.load_state_dict(state_dict["state_dict"])
        logger.info("Loaded model for evaluation.")
        
        # Run evaluation and visualization in test mode
        with torch.no_grad():
            run_evaluation(device, args, model, growth_model, logger)
            run_visualization(device, args, model, logger)
        
        logger.info("Test mode completed.")
        exit()
    else:
        logger.info(model)
        n_param = count_parameters(model)
        logger.info("Number of trainable parameters: {}".format(n_param))

        # Training phase
        train(
            device,
            args,
            model,
            growth_model,
            regularization_coeffs,
            regularization_fns,
            logger,
        )
        
        # Post-training evaluation and visualization (if specified)
        logger.info("Training completed.")
        
        with torch.no_grad():
            # Check if any evaluation is requested
            eval_requested = any([
                getattr(args, 'eval_nll', False),
                getattr(args, 'eval_kantorovich', False),
                getattr(args, 'eval_kantorovich_v2', False),
                getattr(args, 'eval_mse', False),
                getattr(args, 'evaluate_mse_at_timepoints', False),
                getattr(args, 'eval_path_length', False),
                getattr(args, 'generate_eval_samples', False)
            ])
            
            if eval_requested:
                run_evaluation(device, args, model, growth_model, logger)
            
            # Check if any visualization is requested
            viz_requested = any([
                getattr(args, 'plot_vector_fields', False),
                getattr(args, 'plot_trajectories', False),
                getattr(args, 'plot_loss', False)
            ])
            
            if viz_requested:
                run_visualization(device, args, model, logger)

    logger.info("Main execution completed.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)