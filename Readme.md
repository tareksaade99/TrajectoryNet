# Modeling Dynamic Biological Processes with TrajectoryNet

This project explores and reproduces the results of **TrajectoryNet**, a method that connects **Continuous Normalizing Flows (CNFs)** with **Dynamic Optimal Transport (DOT)** to model continuous cellular trajectories from discrete single-cell RNA sequencing (scRNA-seq) snapshots.

In biological research, scRNA-seq data captures cell states at isolated time points, making it difficult to observe continuous developmental processes. TrajectoryNet addresses this by learning a **continuous-time velocity field** that transforms one cell population into another while minimizing transport cost and enforcing biological smoothness.

## Key Concepts
- **Dynamical Optimal Transport (DOT):** Provides a mathematical framework for modeling the continuous movement of probability mass between distributions over time.  
- **Continuous Normalizing Flows (CNFs):** Neural ODE–based generative models that map simple base distributions into complex data distributions through learned continuous transformations.  
- **TrajectoryNet:** Extends CNFs by adding regularizations inspired by optimal transport and biology (energy, Jacobian, and velocity terms) to learn smooth and realistic cellular trajectories.

## Implementation
The implemented model builds upon:
- A neural ODE architecture (`ODEnet`) defining the vector field  
- CNF layers for continuous transformation  
- Regularization terms enforcing smoothness and biological plausibility  
- A backward-integration training strategy using the adjoint method for efficient gradient computation

## Experiments
To validate the approach, the model was tested on **synthetic 2D “arch”-shaped data**, mimicking nonlinear biological trajectories:
- Without regularization, trajectories collapsed to straight lines (low-energy OT behavior)
- With **velocity regularization**, the model successfully reconstructed the arch shape, confirming the benefit of external velocity information

**Results:**  
| Condition | EMD (t₁) | MSE (t₁) |
|------------|-----------|-----------|
| No regularization | 0.5652 | 0.5591 |
| Velocity regularization | 0.3436 | 0.1499 |

## Insights
- Velocity regularization is crucial for recovering realistic non-linear paths  
- Lack of reliable velocity data remains a challenge in real biological contexts  
- Future directions include exploring **non-Euclidean spaces** and **non-quadratic cost functions** to naturally induce curvature without relying on external cues

## Contributions
- Theoretical synthesis of CNFs and DOT foundations  
- Implementation of TrajectoryNet-inspired training with flexible evaluation and visualization tools  
- Quantitative and qualitative experiments demonstrating the effects of regularization on trajectory inference  

This work illustrates how deep generative modeling and optimal transport theory can be combined to infer dynamic biological processes from static data.
