# Optimal Dimensionality Reduction using Conditional Variational AutoEncoder

<a id="readme-top"></a>

This Git repository is associated with the article *Optimal Dimensionality Reduction using Conditional Variational AutoEncoder* available on [TCHES website](https://tches.iacr.org/index.php/TCHES/article/view/12214).

<!-- Table of contents -->
<details>
  <summary>Table of contents</summary>
  <ol>
    <li>
      <a href="#content-of-the-repository">Content of the repository</a>
      <ul>
        <li><a href="#context">Context</a></li>
        <li><a href="#implementation-tricks">Implementation tricks</a></li>
        <li><a href="#repository-structure">Repository structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

## Content of the repository

### Context
We provide in this repository cVAE-OSM implementation and notebooks allowing to re-execute simulations and attacks conducted in our article. 

It should be noted for *Attack.ipynb* notebook that the attack scenario is conducted on simulated traces because we do not provided in this repository the datasets we attacked for environmental reasons, as they are publicly available. 

They can be downloaded here:

- **DPA contest v4.2**: https://dpacontest.telecom-paris.fr/v2 
- **AES_HD_Ext**: https://github.com/AISyLab/AES_HD_Ext 
- **ASCAD v1-F**: https://github.com/ANSSI-FR/ASCAD/blob/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key
- **ASCAD v1-R**: https://github.com/ANSSI-FR/ASCAD/blob/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key
- **SCANTRU**: https://github.com/ANSSI-FR/scantru

Moreover, we draw users' attention to the fact that, to ensure proper attack execution, lines 133 and 139 in *attack.py* file must also be adapted to the targeted dataset.

We developed our model in Python **3.11.8**, using Tensorflow [AAB+15] and Keras [C+15] libraries. 
We recommend to use Python **3.11** versions.

### Implementation tricks

We point out that when implementing cVAE-OSM, a particular attention had to be paid to initialization of encoder weights characterizing $\Sigma_\phi$. 
Indeed, since these weights represent estimated variances at each sample of traces, we initialize all weights characterizing $\Sigma_\phi$ to 1 and add a custom constraint that forces weights during cVAE-OSM training to be always positive.
It is important to take this specificity into account during implementation to ensure proper autoencoder working. 
We thus consider this type of initialization and update for $\Sigma_\phi$. 
We do not investigate impact of initialization and weights constraints on cVAE-OSM, especially on its weights convergence. 
This investigation should be part of a future work. 

Since we consider that the basis used to describe the deterministic part $\Psi$ includes a bias term and that the optimal dimensionality reduction does not involve it (see Theorem 2), we implement our model in such a way as to remove biases included in dense layers. 

Finally, as a relationship between the variance $\sigma^2_\phi$ and mean $\mu_\phi$ of monovariate traces $\mathbf{\tilde{T}}$ is defined *i.e.* $\mu_\phi$ (resp. $\sigma^2_\phi$) must converge towards $D$ (resp. $2D$) (see Section 3.3), we decide to create a custom dense layer for $\sigma^2_\phi$ computation.
It consists in estimating the weights related to $\mu_\phi$ and then, use those estimations to compute $\sigma^2_\phi$ instead of re-estimating them.
Considering $D$ as the dimension of traces, this trick therefore reduces the number of trainable parameters by $D$ compared with the expected theoretical complexity defined in Section 3.3 (paragraph Neural network complexity).
Hence, this allows us to achieve the final architecture complexity presented in Proposition 1.

<a id="cvae-picture"></a>
<p align="center">
  <a href=""><img src="cvae_picture.svg" title="more details in cVAE-OSM article." alt="cVAE-OSM article" style="width:1000px;height:600px;"></a>
</p>
<p align="center">cVAE-OSM architecture.</p>

### Repository structure

Our repository has the following structure:
```bash
.
|   Attack.ipynb
|   cvae_picture.svg
|   Experiment_1.ipynb
|   Experiment_2.ipynb
|   Experiment_3.ipynb
|   Experiment_4.ipynb
|   poetry.lock
|   pyproject.toml
|   requirements.txt
|
└── cvae_osm_utils
        attack.py
        cVAE_OSM_model.py
        cVAE_OSM_tools.py
        experiments_tools.py
        generate_traces.py
        Kernel_Weights_Constraints.py
        __init__.py       
```
This repository contains 5 notebooks, 3 files, a picture and a package which includes 6 modules.

In the following, we briefly summarize the contents of each file.

As previously explained, these notebooks allow users to re-execute simulations and attacks conducted in Section 5.
- *Attack.ipynb* is a notebook in which we carry out profiled attacks using cVAE-OSM and following stategy provided in Section 4.2.
- *cvae_picture.svg* is a <a href="#cvae-picture">picture of cVAE-OSM architecture</a>.
- *Experiment_1.ipynb* allow users to reproduce the experiment on simulations about leakage model and variance extraction that is depicted in Section 5.1.2.
- *Experiment_2.ipynb* reproduces the experiment on simulations provided in Section 5.1.3, which is about the optimal dimensionality reduction performed by cVAE-OSM.
- *Experiment_3.ipynb* reproduces the experiment carried out in Section 5.1.4, which assess cVAE-OSM ability to overcome Small Sample Size (SSS) or High-Dimension Low Sample Size (HDLSS) problem.
- *Experiment_4.ipynb* includes all experiences depicted in Section 5.1.5, about the practical issues. 
- *poetry.lock*, *pyproject.toml* and *requirements.txt* files are described in Section <a href="#getting-started">Getting started</a>.

In addition, we provide a package called $`\texttt{cvae\_osm\_utils}`$ that contains modules necessary for notebooks running.

- *attack.py* implements the profiled attack strategy introduced in Section 4.2.
- *cVAE_OSM_model.py* implements cVAE-OSM model.
- *cVAE_OSM_tools.py* includes all auxiliary functions that can be useful when using cVAE-OSM such as weights visualization function or projection onto the Guilley *et al.* orthonormal basis [GHMR17] that is used in the paper. 
- *experiments_tools.py* includes all functions necessary to reproduce our experiments on simulations. 
- *generate_traces.py* implements a trace generation function.
- *Kernel_Weights_Constraints.py* implements custom weights constraint explained in <a href="#implementation-tricks">Implementation tricks</a> section.
- *\_\_init\_\_.py* empty file required to create our $`\texttt{cvae\_osm\_utils}`$ package.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started


### Prerequisites

To enforce experiment reproducibility, we suggest the use of poetry tool (https://python-poetry.org/).
We also provide a `requirements.txt` file to reproduce the Python environment used to perform experiments with `pip install`.

In case both solutions are not suited (impossibility of using a virtual environment) we alternatively provide a list of dependencies with no version information.
In this later case there is a high probability of not being able to reproduce the same results and/or being forced to adapt part of the code.

To use the solution based on Poetry it must be installed following the [install instructions](https://python-poetry.org/docs/#installation).

### Installation
We list here the different setup techniques ordered by decreasing reproducibility. We recommend to test the first one then switch to the next in case of failure and so on.

<a id="install-1"></a>
#### 1. Using Poetry with Provided `poetry.lock`

From the git root directory (where this readme file is), run

    poetry install
    
It will use the `poetry.lock` file to replicate the environment used for the paper.

If the installation succeeded, you can now launch the virtual environment using the command:

    poetry shell
  
or

    poetry env activate
  
depending on the poetry version used.

Note that you can alternatively source the `activate` file from the environment.

> **Troubleshooting.**
>
> An error may arise when some package version is not available yet/anymore for a given Python version.
> In that case, we recommend first trying to use the same Python version as used by the authors *i.e.* Python 3.11.

#### 2. Using a Docker Container and `requirements.txt`
To run the same version as authors, a simple possibility is to use a docker container running the python version with jupyter.

    sudo docker run -it --rm -p 8888:8888 -v .:/home/jovyan --user root -e GRANT_SUDO=yes jupyter/base-notebook:x86_64-python-3.11.5

**Note.**
The `--user root -e GRANT_SUDO=yes` part of the command is required to install texlive package.

Clicking on the link shown in the terminal opens a jupyterlab webpage. Then, opening a terminal from this interface, packages
can be installed using the `requirements.txt` file which contains the precise versions of packages used by the authors.
For plots, latex installation is also required (`texlive-latex-extra` package and its dependencies). 

    pip install -r requirements.txt
    sudo apt update && sudo apt install cm-super texlive-latex-extra

> **Troubleshooting.**
>
> In case of troubles with the embedded terminal, an alternative is to use a notebook to execute this command.
> To do so, the following content must be executed in a first cell:
>
>     %%bash
>     sudo apt update
>     DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC sudo apt install -y tzdata
>
> And the following content in a second cell:
>
>     %%bash
>     sudo apt install -y cm-super texlive-latex-extra dvipng
>     pip install -r requirements.txt


> **Known issues.**
>
> In cases where everything runs correctly within the Docker environment, but an error still occurs when attempting to visualize the weights, it is recommended to comment out line 136 in the *cVAE_OSM_tools.py* file in $`\texttt{cvae\_osm\_utils}`$ module (`weights_visualization` function) to bypass the issue.
Once the modification is made, restart the notebook to apply the changes.


#### 3. Using Poetry without Provided `poetry.lock`
If the previous solutions did not work or are not suitable, the next alternative is to use Poetry without `poetry.lock`.
In that case, installed packages will have versions at least as high as the one used by authors with the same major version (first version digit). This provides a bit more flexibility while still maintaining a high probability of reproductibility.

For this, the `poetry.lock` must be adapted to the current Python version.

    poetry lock
    poetry install

First a new lockfile is computed, then packages are installed.
Same commands as for <a href="#install-1">installation 1</a> can be used to activate the environment.

#### 4. Using `requirements.txt`

If you are not able to install/run Poetry and/or Docker container without error, then you can create a new virtual environment with the classical `python` command:

    python -m venv .venv

Then activate the environment and install the dependencies:

    source .venv/bin/activate
    pip install -r requirements.txt

> <u>Caution</u>: To avoid troubles, we recommend to use the same Python version as the one used by the authors, *i.e.* Python 3.11.  

#### 5. Using Dependency List

Ultimately, in case the Python version is too different from the one used by authors, some package may not be available with
an acceptable version.

In that later case the only solution to run the experiments is to install dependencies without taking care of the versions.
It is recommended to create a virtual environment but one might install the corresponding packages at the system level if required.

    python3 -m venv .venv --prompt cVAE-prompt
    source .venv/bin/activate
    pip install tensorflow scipy numpy scikit-learn matplotlib ipykernel tqdm

> Warning! The reproducibility of the results is then not guaranteed at all.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation

If you use our code, model or wish to refer to our results, please use the following BibTex entry:
```
@article{Boussam_Carbone_Gérard_Renault_Zaid_2025,
title={Optimal Dimensionality Reduction using Conditional Variational AutoEncoder},
volume={2025},
url={https://tches.iacr.org/index.php/TCHES/article/view/12214},
DOI={10.46586/tches.v2025.i3.164-211},
number={3},
journal={IACR Transactions on Cryptographic Hardware and Embedded Systems},
author={Boussam, Sana and Carbone, Mathieu and Gérard, Benoît and Renault, Guénaël and Zaid, Gabriel},
year={2025},
month={Jun.},
pages={164–211} 
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## References

[AAB+15] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[C+15] François Chollet et al. Keras. https://keras.io, 2015.

[GHMR17]  Sylvain Guilley, Annelie Heuser, Tang Ming, and Olivier Rioul. Stochastic side-channel leakage analysis via orthonormal decomposition. In Innovative Security Solutions for Information Technology and Communications: 10th International Conference, SecITC 2017, Bucharest, Romania, June 8–9, 2017, Revised Selected Papers 10, pages 12–27. Springer, 2017.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
