# Symmetries and Synchronization from Whole-Neural Activity in *C. elegans* Connectome

This repository accompanies the paper **"Symmetries and synchronization from whole-neural activity in *C. elegans* connectome: Integration of functional and structural networks."** It contains the Python code and analysis scripts used to explore the relationship between network symmetries and synchronization in the nervous system of *C. elegans*.

## Abstract

Understanding the dynamical behavior of complex systems from their underlying network architectures is a long-standing question in complexity theory. Many metrics, such as motifs, centrality, and modularity, have been used to characterize network features. Network symmetries, however, hold particular importance due to their role in underpinning the synchronization of system units—a common feature in nervous system activity patterns.

In this study, we present a principled method for inferring network symmetries by integrating functional neuronal activity data with structural connectome information. Using nervous system-wide population activity recordings of the *C. elegans* backward locomotor system, we identify and analyze fibration symmetries in the connectome. These structures provide insights into neuron groups that synchronize their activity and reveal functional building blocks in the motor periphery.

This repository contains the code used to analyze these symmetries, offering a computational framework to investigate structure-function relationships in biological networks. Our approach provides a foundation for further studies in complex systems, including the nervous systems of larger organisms.

---

## Repository Contents

- **`connectomes/`**  
  Contains Varshney C. Elegans connectomes used in the study, including:
  - Uncollapsed Varshney connectome
  - Collapsed Varshney connectome
  
- **`colorings/`**  
  Contains generated colorings with different settings, more can be generated using the code.

- **`README.md`**  
  This file, providing an overview of the repository.

---

## Key Features

1. **Fibration Symmetry Detection:** Implements algorithms to identify fibration symmetries in the *C. elegans* connectome, linking structural features to functional synchronization.
2. **Integration of Functional and Structural Data:** Combines connectome data with functional activity recordings to provide a holistic view of the nervous system's dynamics.
3. **Reproducible Analysis Pipeline:** Modular codebase to preprocess data, detect symmetries, and validate findings with statistical tools.
4. **Visualization Tools:** Includes scripts to generate publication-quality figures of network structures, synchronization patterns, and functional groups.

---

## How to Use

### Prerequisites

- Python 3.8 or higher
- Required Python libraries (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

### Running the Analysis

1. Clone this repository:
   ```bash
   git clone https://github.com/ialireza13/c_elegans_neural_network.git
   cd c_elegans_neural_network
   ```
2. Load and explore the data using the `handson.ipynb` Jupyter notebook.
3. Run the core analysis pipeline and view results in the notebook, also saved in the directory.

<!-- ---

## Citation

If you find this code useful in your research, please cite:

> Author(s). (Year). **Symmetries and synchronization from whole-neural activity in *C. elegans* connectome: Integration of functional and structural networks.** *Journal/Conference Name*.

--- -->

## Acknowledgments

This work is inspired by and builds upon existing tools and concepts in network science, graph theory, and neuroscience. We acknowledge contributions from collaborators and funding sources supporting this research.

---

This repository is not designed as a standalone software package but serves as a record and resource for reproducing and extending the findings of the associated publication. For questions or collaborations, please contact the authors.