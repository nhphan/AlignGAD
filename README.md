# AlignGAD

Official repository for **AlignGAD**, accepted at **The Pacific Rim
International Conference on Artificial Intelligence (PRICAI) 2026**.

AlignGAD studies zero-shot generalized graph anomaly detection: the model learns
from source graphs and is transferred to unseen target graphs without
target-domain training. The central idea is to make node reconstruction more
transferable by aligning graph signals across domains, constructing
cluster-aware graph views, and calibrating node-level discrepancy scores with
source supervision.

## Overview

![AlignGAD architecture](fig/main_structure.png)

At a high level, AlignGAD combines:

- Global graph signal unification
- Cluster-aware multi-view graph construction
- Node reconstruction and discrepancy scoring
- Source-guided score calibration
- Multi-view anomaly score aggregation

## Results

![AlignGAD results](fig/results.png)

The figure above summarizes the reported cross-domain anomaly detection results.
For exact experimental settings, source/target splits, and discussion, please
refer to the paper.

## Code

The core reference implementation is provided in:

```text
aligngad.py
```

Install the basic dependencies:

```bash
pip install -r requirements.txt
```

Then adapt `load_graph(name)` so that it returns:

```python
A, X, y
```

where `A` is the adjacency matrix, `X` is the node feature matrix, and `y` is the
binary anomaly label vector when labels are available.

After defining your loader, a typical run looks like:

```bash
python aligngad.py \
  --sources Facebook Flickr BlogCatalog ACM \
  --targets Cora Citeseer Pubmed Photo CS Amazon YelpChi Reddit
```

## Citation

```bibtex
@misc{nguyen2026zeroshotgeneralizedgraphanomaly,
  title={A Zero-shot Generalized Graph Anomaly Detection Framework via Node Reconstruction},
  author={Phan Nguyen and Dat Cao and Hien Chu and Khue Hoang},
  year={2026},
  eprint={2606.12673},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.12673},
}
```
