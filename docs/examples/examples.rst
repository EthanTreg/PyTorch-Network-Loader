.. _examples:

********
Examples
********

Here we show some real world examples that use PyTorch Network Loader.

See the
`Jupyter Notebook <https://github.com/EthanTreg/PyTorch-Network-Loader/blob/master/notebooks/training_example.ipynb>`_
for a short demo.

Variational Autoencoder with Normalizing Flow for X-Ray Spectral Fitting
========================================================================

| Authors: F. Redmen, E. Tregidga, J. Steiner, C. Garraffo
| Date: January 2026
| GitHub: `fspnet-var <https://github.com/fi-redmen/fspnet-var>`_
| Peer-Reviewed Paper: `Machine Learning and the Physical Sciences <https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_365.pdf>`_
| ArXiv: `arXiv 2601.07440 <https://arxiv.org/abs/2601.07440>`_
| NetLoader Version: 3.10.1

They introduce a probabilistic model, utilizing a variational autoencoder with a normalizing flow, trained to adopt a
physical latent space.
This neural network produces predictions for spectral-model parameters as well as their full probability distributions.
Their implementations result in a significant improvement in spectral reconstructions over a previous deterministic
model while performing three orders of magnitude faster than traditional methods.

Measuring the Dark Matter Self-Interaction Cross-Section with Deep Compact Clustering for Robust Machine Learning Inference
===========================================================================================================================

| Authors: E. Tregidga, D. Harvey, L. Biggio, and F. Vecchi
| Date: November 2025
| GitHub: `Bayesian-DARKSKIES <https://github.com/EthanTreg/Bayesian-DARKSKIES>`_
| Peer-Reviewed Paper: `Astronomy & Astrophysics <https://doi.org/10.1051/0004-6361/202556629>`_
| ArXiv: `arXiv 2511.09660 <https://arxiv.org/abs/2511.09660>`_
| NetLoader Version: 3.9.4

This project implements a clustering neural network for constraining the dark matter self-interaction cross-section
from galaxy clusters using cosmological simulations.
The method builds an interpretable latent space where galaxy clusters are grouped by physical similarity, enabling
robust parameter estimation in sparsely sampled parameter spaces while providing confidence measurements to detect
out-of-domain data and exploration of secondary features in the data.
They apply this method to the BAHAMAS-SIDM and DARKSKIES simulations and demonstrate accurate recovery of the
self-interaction cross-section when the test data lies within the training domain and reliable rejection of foreign
datasets, offering a blueprint for robust machine learning for scientific inference.

Rapid Spectral Parameter Prediction for Black Hole X-Ray Binaries using Physicalized Autoencoders
=================================================================================================

| Authors: E. Tregidga, J. Steiner, C. Garraffo, C. Rhea, and M. Aubin
| Date: October 2023
| GitHub: `Spectrum-Machine-Learning <https://github.com/EthanTreg/Spectrum-Machine-Learning>`_
| Peer-Reviewed Paper: `Monthly Notices of the Royal Astronomical Society <https://doi.org/10.1093/mnras/stae629>`_
| ArXiv: `arXiv 2310.17249 <https://arxiv.org/abs/2310.17249>`_
| NetLoader Version: 3.2.4

This project presents a new semi-supervised autoencoder neural network for parameter prediction and spectral
reconstruction of black hole X-ray binaries (BHBs), showing an improvement of up to a factor of 2,700 in speed, while
maintaining comparable accuracy.
The approach maps the spectral features from the numerous outbursts catalogued by NICER and generalises them to new
systems for efficient and accurate spectral fitting.
The effectiveness of this approach is demonstrated in the spectral fitting of BHBs and holds promise for use in other
areas of astronomy and physics for categorising large dataset.

Simulation-Based Inference for Exoplanet Atmospheric Retrieval: Insights from Winning the Ariel Data Challenge 2023 Using Normalizing Flows
===========================================================================================================================================

| Authors: M. Aubin, C. Cuesta-Lazaro, E. Tregidga, J. Viaña, C. Garraffo, I. Gordon, M. López-Morales, R. Hargreaves, V. Makhnev, J. Drake, D. Finkbeiner, and P. Cargile
| Date: September 2023
| GitHub: `Ariel_Data_Challenge_2023_solution <https://github.com/AstroAI-CfA/Ariel_Data_Challenge_2023_solution>`_
| Peer-Reviewed Paper: `Springer Nature <https://doi.org/10.1007/978-3-031-74643-7_10>`_
| ArXiv: `arXiv 2309.09337 <https://arxiv.org/abs/2309.09337>`_
| NetLoader Version: Pre-Release

This project presents novel machine learning models developed by the `AstroAI <https://astroai.cfa.harvard.edu/>`_ team
for the `Ariel Data Challenge 2023 <https://www.ariel-datachallenge.space/>`_, where one of the models secured the top
position among 293 competitors.
Leveraging Normalizing Flows, their models predict the posterior probability distribution of atmospheric parameters
under different atmospheric assumptions.
Moreover, they introduce an alternative model that exhibits higher performance potential than the winning model, despite
scoring lower in the challenge.
These findings highlight the need to reevaluate the evaluation metric and prompt further exploration of more efficient
and accurate approaches for exoplanet atmosphere spectra analysis.
Finally, they present recommendations to enhance the challenge and models, providing valuable insights for future
applications on real observational data. These advancements pave the way for more effective and timely analysis of
exoplanet atmospheric properties, advancing our understanding of these distant worlds.
