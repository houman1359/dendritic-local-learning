# DANDI fluorescence wording audit

Use **supplied ROI fluorescence traces**, without calling them raw or dF/F.

Suggested Methods sentence:

> Trial responses were the mean of the supplied ROI fluorescence traces in `RoiResponseSeries` over each stimulus interval; our analysis did not deconvolve the traces or apply a reliability correction.

The extractor reads the supplied `data` arrays directly in `code/task_derived/microns_dandi_nwb_trial_extract.py:291`. Its `interval_means` function selects timestamps between stimulus start and stop, including endpoints, and computes `numpy.nanmean` across those samples (`:215–227`). The calling wrapper sets both offsets to zero (`code/task_derived/extract_functional_partner_responses.py:102–103`). These operations establish what this analysis did, rather than what upstream processing produced the supplied traces.

The retained scan extract summary identifies asset `50b43c75-686f-4d06-acf2-cd0b1b42e8be`, `sub-17797/sub-17797_ses-4-scan-10_behavior+image+ophys.nwb`. The original summary is `../../dendritic-credit-routing/results/microns_functional_partner_responses/target256456_ses4_scan10_automatic_conservative/microns_dandi_trial_extract_summary.json` relative to the journal. The DANDI API identifies its SHA256 as `ea33de99377b1547cffa94900a1f54239ec39d67f3d991ec3afde8da35271ad1`; this upstream digest was not recomputed locally.

A direct remote HDF5 metadata inspection of this asset is retained in `dandi_fluorescence_metadata.json`. The `/processing/ophys` description is “processed 2p data.” All eight series are described as fluorescence traces, with unit `n.a.`, conversion 1 and offset 0. The metadata does not establish dF/F normalization, raw image-derived intensities, or the absence of prior signal processing. No trace or image arrays were downloaded during this audit.

The public [MICrONS-to-NWB exporter](https://github.com/catalystneuro/MICrONS-to-nwb/blob/main/src/microns_to_nwb/tools/ophys/ophys.py) reads the upstream `nda.Fluorescence` table's `trace` entries directly into `RoiResponseSeries`; it does not add a raw or dF/F designation. The [DANDI MICrONS tutorial](https://docs.dandiarchive.org/example-notebooks/000402/MICrONS/demo/000402_microns_demo/#fluorescence-traces) likewise describes supplied fluorescence traces within the processing module. These sources support the proposed neutral term. A claim about a particular upstream preprocessing pipeline would require further provenance, and is unnecessary here.

At inspection, unsupported uses of “raw” appeared in main Methods, two SI sentences and `code/task_derived/README.md`. No manuscript, supplement or loader files were modified by this audit.
