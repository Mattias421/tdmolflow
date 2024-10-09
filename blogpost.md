# Molecule generation with trans-dimensional flow-matching


Molecules are important for drug and material discovery. Being able to model the distribution of 3D molecules is helpful tool to find desirable molecules and inference their likelihood. The highest quality methods to date involve *conditional flow-matching*, which learns to interpolate random-noise samples into molecule samples. Although this method works well, the number of atoms in the molecule to generate must be known at inference time, removing an important factor of molecule generation from the flow-matching process. This problem has been addressed for similar yet inferior *diffusion* models in the form of *Trans-Dimensional Generative Modeling via Jump Diffusion Models*, which model both state and dimension within the same framework. This leads to the question: can we extend the benefits of jump-diffusion models to flow-matching models?
This serves as the foundations for conditional generation which could be used to further assist molecule discovery e.g. generating molecules that match a given text description. 


**TL:DR:** Generating molecules is better when jointly modelling atom state and number of atoms, but this hasn't been applied to SOTA methods (molecule flow-matching). Doing so would lead to a solid foundation for more complex models e.g. text-to-mol.

## Flow-matching
![FM between 1D distributions](https://mlg.eng.cam.ac.uk/blog/assets/images/flow-matching/representative.gif)
Continuous normalizing flows (CNFs) learn velocity fields that transport samples from a normal distribution to a given data distribution. Score-based diffusion modelling is an example of a method to train CNFs, but fails to learn stable probability dynamics and has inefficient sampling speed. On the other hand, flow-matching is an efficient method to train CNFs with faster inference and stable training dynamics. The GIF above ([from Fjelde et al.'s blog](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html#but-is-cfm-really-all-rainbows-and-unicorns)) shows the flow between a normal distribution and a more complex Gaussian.

## Trans-dimensional modelling
![jump diffusion from Campbell et al.](https://github.com/andrew-cr/jump-diffusion/raw/main/molecules/assets/image_genprog.png)[Molecule image from [Campbell et al.](https://github.com/andrew-cr/jump-diffusion/tree/main/molecules)]
One key issue with molecule generation is that the number of atoms a molecule has affects the properties and nature of that molecule. A simple solution is to require the desired number of atoms, or predict with a simple method, but these are unfavourable as this splits the topology of our molecule distribution by number of atoms. Trans-dimensional jump-diffusion ([Campbell et al.](https://github.com/andrew-cr/jump-diffusion/tree/main/molecules)) jointly models state (e.g. atom positions) and dimension (number of atoms in the molecule) with a denoising process that starts with a single "noisy" atom and iteratively adds new atoms and perturbs atoms according to a jump-diffusion process to generate molecules. Although this method addresses the trans-dimensional nature of molecule generation, it has the same shortcomings as diffusion, leading to the natural question of whether trans-dimensional flow-matching is possible.

## Trans-dimensional flow-matching
To the best of our knowledge, *trans-dimensional flow-matching* has not been formulated yet. This project aims to combine the advantages of flow-matching and trans-dimensional diffusion. This is not trivial because although diffusion models learn the reverse of a noising-process, flow-matching models learn the interpolation between a normal distribution and a molecule distribution. This stark difference to diffusion means that we cannot simply remove atoms from our molecule then add noise to the remainder, we must instead find the optimal interpolation between a noisy atom and a molecule. In theory, this is the same as interpolating between a noisy atom and it's optimal transport to the desired molecule. If we interlace the perturbation steps with "jumps" (randomly adding new atoms until satisfied), and compute a optimal transport for the new interpolant (noisy molecule), the interpolating path will be optimal and act as a push-forward map between the single noisy atom distribution and the molecule data.

## Roadmap
- Devise trans-dimensional flows e.g. through optimal transport potentials
- Study code for [jump-diffusion](https://github.com/andrew-cr/jump-diffusion/tree/main/molecules) and molecule [flow-matching](https://github.com/AlgoMole/MolFM)
- Implement and train an unconditional trans-dimensional flow-matching model
- (Make framework more efficient if need be)
- Extend model to simple conditional tasks e.g. property
- Then extend to more complex conditioning e.g. text via cross-attention

## Relevant articles
[Scale optimal transport (SemlaFlow) by AstraZeneca](https://arxiv.org/pdf/2406.07266)
[Flow matching blog](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html#quick-summary)
[EquiFM](https://arxiv.org/pdf/2312.07168)

## Relevant code
https://github.com/AlgoMole/MolFM
https://github.com/andrew-cr/jump-diffusion/tree/main/molecules
https://github.com/atong01/conditional-flow-matching/tree/main

Some jax libraries good for learning how neural differential equations and optimal transport work
https://github.com/patrick-kidger/diffrax/releases
https://github.com/ott-jax/ott
