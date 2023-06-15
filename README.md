# Pathfinder

Goal of this experimental project is to build deep learning based search mechanism to efficiently find small molecule binders for given protein. Main hypothesis behind this project is that with correct [embeddings](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526) for both small molecules and protein structures, it's possible to train a model that would allow to translate them to the same embeddings space.

If we could build such model, finding molecules for proteins would be simple task of finding most similar vectors, which would be very scalable and efficient way of screening enormous libraries of molecules.

# Architecture

Entire project will consist of 3 major components.

## Protein embedding model

First step is to train an embedding representation of a protein. There is some prior literature on this like [dMASIF](https://github.com/FreyrS/dMaSIF). There are some failed experiments in `protein-embeddings` dir with graph nns and 2d conv nets of adjecency matrices.

Current approach is to use [GearNet](https://github.com/DeepGraphLearning/GearNet) and their pre-trained models trained on Alphafold2 predicted structures as basic embeddings for protein structures. Paper related to this code has also good explanation of tasks and evaluation strategies for embeddings of protein structures.

Useful datasets:
* [Alphafold predicted structures](https://alphafold.ebi.ac.uk/)

## Molecule embedding model

Similarly to protein embeddings, we should calculate embeddings for small molecules.

Unlike proteins, molecule embeddings should require conformation (or 3d structure) of molecules, as number of rotatable bonds can quickly make any search problem intangible (billions of molecules ** thousands or more of conformations). 

Some approaches for mol embeddings could be language models (based on [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) or [SMARTS](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html) representations) or graph based models.

There are few available datasets with unlabeled molecules. Typically from molecule vendors, but also academia.

Datasets:
* [Chembl](https://www.ebi.ac.uk/chembl/) is featurized dataset of 2.4M compounds with some features, including, in some cases, binding target proteins and their [pChembl](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions#what-is-pchembl) value (good measure of binding affinity for molecules)
* [Enamine](https://enamine.net/compound-collections/real-compounds/real-database) - around 6M compounds available for purchase from Enamine
* [ZINC15](https://zinc15.docking.org/substances/subsets/) - huge dataset of 2B mols spread into smalled datasets

## Embedding translator

This is new model that we have to train. This would use smaller dataset of bound molecules to train neural network that would translate protein embeddings to molecule embeddings. Current idea is to use contrastive learning and multimodal learning to train such embedding translator. This is new and largery unproven approach.

Recent development of multimodal architectures (text-to-image, image search etc) is good inspiration. Especially field of image search.

Datasets to train these models would need a protein pose with docked molecule. There are few available

* [BigBind](https://bigbind.mml.unc.edu/)
* PDBBind - TODO: provide good download link
* [scPDB](http://bioinfo-pharma.u-strasbg.fr/scPDB/)