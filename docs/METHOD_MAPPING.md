# Method Mapping

This repository maps the paper components to code as follows.

| Paper component | Code |
|---|---|
| BF+MF guide generation | `ngps/guide.py` |
| Direction-aware mask | `make_misalignment_mask` in `ngps/guide.py` |
| Guide patch SSD search | `NeighborGuidedPatchSampler._ssd_stack` |
| Top-K raw center-pixel retrieval | `NeighborGuidedPatchSampler.forward_direction` |
| Static N2N reconstruction | `NGPSHybridLoss` in `ngps/losses.py` |
| NGPS dynamic reconstruction | `NGPSHybridLoss` in `ngps/losses.py` |
| Regional consistency | `NGPSHybridLoss` in `ngps/losses.py` |
| Same-backbone NS-N2N-style baseline | `NSN2NLoss` and `scripts/train_nsn2n.py` |
| Denoiser backbone | `ngps/nafnet.py` |

At inference time, NGPS uses only the trained denoiser. Guide generation,
patch search, and Top-K retrieval are training-time target construction steps.
