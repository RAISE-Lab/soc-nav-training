# Improving Generalization in Reinforcement Learning Training Regimes for Social Robot Navigation

Our [Paper]([url](https://arxiv.org/pdf/2308.14947)) was accepted at the [Generalization in Planning]([url](https://aair-lab.github.io/genplan23/)) workshop at NeurIPS 2023!



## Abstract

In order for autonomous mobile robots to navigate in human spaces, they must abide by our social norms. Reinforcement learning (RL) has emerged as an effective method to train sequential decision-making policies that are able to respect these norms. However, a large portion of existing work in the field conducts both RL training and testing in simplistic environments. This limits the generalization potential of these models to unseen environments, and the meaningfulness of their reported results. We propose a method to improve the generalization performance of RL social navigation methods using curriculum learning. By employing multiple environment types and by modeling pedestrians using multiple dynamics models, we are able to progressively diversify and escalate difficulty in training. Our results show that the use of curriculum learning in training can be used to achieve better generalization performance than previous training methods. We also show that results presented in many existing state-of-the-art RL social navigation works do not evaluate their methods outside of their training environments, and thus do not reflect their policies' failure to adequately generalize to out-of-distribution scenarios. In response, we validate our training approach on larger and more crowded testing environments than those used in training, allowing for more meaningful measurements of model performance.



## External Code Usage

<!-- ### Files Changed

The novel work in this project is mainly in `CrowdNav/crowd_nav/`. Major modifications were made to `CrowdNav/crowd_sim/crowd_sim.py`, and minor modifications were made to files in `CrowdNav/crowd_sim/utils/`. Succinct, but crucial modifications were made in `CrowdNav/crowd_sim/policy/` and especially in `CrowdNav/crowd_sim/policy/socialforce/`. -->

External code is included in this repository. This was done to streamline adherence to McGill's evaluation process. The novel work in this project is built upon **Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning** [[Paper](https://arxiv.org/abs/1809.08835)], [[GitHub](https://github.com/vita-epfl/CrowdNav)]. The bulk of this work was done in the `CrowdNav/crowd_sim/`.

Additionally, this repository contains Python-RVO2 (`soc-nav-training/Python-RVO2/`) [[GitHub](https://github.com/sybrenstuvel/Python-RVO2)], the official implementation of ORCA [[Paper](https://gamma.cs.unc.edu/ORCA/publications/ORCA.pdf)], used in Chen et al.'s original work. This repository also contains code based on Deep Social Force (`soc-nav-training/CrowdNav/crowd_sim/envs/policy/socialforce/`) [[Paper](https://arxiv.org/abs/2109.12081)], [[GitHub](https://github.com/svenkreiss/socialforce)] (version 0.1.0).
<!-- It would have been a fork from their repository, but technical difficulties caused me to create a separate one containing mainly their code, but also others.  -->


## Setup
The code in this repository is based on Python 3.6. It is recommended to create a virtualenv. The following steps were tested in Ubuntu 20.04.

1. Download python 3.6 and the corresponding `pip` and `virtualenv` releases if your distribution does not have them already.
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6 python3.6-dev python3.6-venv
sudo python3.6 -m pip install virtualenv
```

2. Navigate to the repository and activate venv.
```
cd /path/to/soc-nav-training/
virtualenv --python=/usr/bin/python3.6 venv
source venv/bin/activate
```

3. Install RVO2 (ORCA). If you run into problems, refer to the [repo](https://github.com/sybrenstuvel/Python-RVO2).
```
cd Python-RVO2/
pip install -r requirements.txt
python setup.py build
python setup.py install
```

4. Install CrowdNav.
```
cd .. # navigate back to main soc-nav-training directory
pip install -r requirements.txt
cd CrowdNav
yes | pip install -e .
```

This should work, but it is possible for issues to arise. If you are unable get it working, please email me.


## Getting Started
This repository is organized in two parts: CrowdNav/crowd_sim/ folder contains the simulation environment and
CrowdNav/crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found [here](CrowdNav/crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed inside the crowd_nav/ folder. Note that in this repo 3 pretrained models are included: SARL (baseline), SARL (diverse), and SARL (curriculum+diverse)

From the **main `soc-nav-training` directory**, navigate to the crowd_nav folder.
```
cd CrowdNav/crowd_nav/
```

Optional: Train a policy in a diverse environment (will take 10+ hours).
```
python train.py --policy sarl --env_config configs/env_multi-test_bl-cr.config
```

Test a policy over 50 test episodes with pretrained models.
```
python test.py --policy orca --phase test --env_config configs/env_multi-test_bl-sq.config
python test.py --policy sarl --model_dir data/sarl_curr_div --phase test --env_config configs/env_multi-test_dn-sq.config
```
Run a pretrained model for one episode and visualize the result.
```
python test.py --policy sarl --model_dir data/sarl_curr_div --phase test --env_config configs/env_multi-test_lg-cr.config --visualize --test_case 0
```


## Simulation Videos (SARL (baseline) in diverse environment)
Baseline Circle Crossing             | Baseline Square Crossing
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/8Ru0I1u.gif" width="400" />|<img src="https://i.imgur.com/zwRfDBB.gif" width="400" />
**Large Circle Crossing**             | **Large Square Crossing**
<img src="https://i.imgur.com/Rh88H52.gif" width="400" />|<img src="https://i.imgur.com/ensprho.gif" width="400" />
**Dense Circle Crossing**             | **Dense Square Crossing**
<img src="https://i.imgur.com/SfKSlXZ.gif" width="400" />|<img src="https://i.imgur.com/D4453gj.gif" width="400" />  |  

## Collision Case in Dense Square Crossing
SARL (baseline) (Collision)             | SARL (curriculum+diverse) (Avoids Collision)
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/4AvnPd1.gif" width="400" />|<img src="https://i.imgur.com/d7l7rNI.gif" width="400" />

## Learning Curve
Learning curve comparison between different training methods.

<img src="https://i.imgur.com/QHpUPMl.png" width="600" />
