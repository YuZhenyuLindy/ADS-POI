# Upstream methods and data

This package implements ADS-POI and local adaptations of sequential recommendation baselines. It does not bundle upstream datasets, model weights or third-party source repositories.

- **GRU4Rec:** *Session-based Recommendations with Recurrent Neural Networks*.
- **SASRec:** *Self-Attentive Sequential Recommendation*.
- **Data preparation:** The input layout follows the [LLM4POI preprocessing repository](https://github.com/neolifer/LLM4POI), which builds on STHGCN and GETNext preprocessing for the relevant datasets.

NumPy, pandas, SciPy, PyTorch and pytest are installed separately and retain their respective licenses. Dataset access and use remain subject to the upstream terms. No new license is assigned to third-party materials by this repository.
