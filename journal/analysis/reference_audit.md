# Reference metadata audit

Date: 4 August 2026

## Scope and method

All 41 records in `references.bib` were checked against the publisher page or official proceedings record and, for DOI-bearing works, the Crossref registration. PubMed/Europe PMC was used as a secondary check for biomedical records and the current version of the Kornfeld preprint. Official arXiv Atom metadata was used for the two author preprints. Google Scholar was not treated as the canonical metadata source because its records can merge versions and inherit user-supplied fields.

BibTeX was run after the edits. It parsed the database with no warnings or errors.

## Corrections made

| Key | Correction | Authoritative source |
|---|---|---|
| `koch1999biophysics` | Added the book DOI. The year remains 1999: Oxford lists an online publication date in November 1998, while the title-page copyright and standard book citation use 1999. | [Oxford Academic](https://academic.oup.com/book/40820) |
| `london2005dendritic`, `branco2010single` | Corrected the BibTeX encoding of Haeusser's umlaut. The underlying author metadata was already correct. | [Crossref: London and Haeusser](https://api.crossref.org/works/10.1146/annurev.neuro.28.061604.135703); [Crossref: Branco and Haeusser](https://api.crossref.org/works/10.1016/j.conb.2010.07.009) |
| `lovettbarron2012regulation` | Corrected three given names: Peter H. Lee, Frederic Bolze, and Xiao-Hua Sun; repaired the cedilla in Jean-Francois Nicoud. | [Nature Neuroscience](https://www.nature.com/articles/nn.3024) |
| `sacramento2018dendritic` | Corrected the tilde in Joao and followed the official proceedings form `Ponte Costa, Rui`. The official NeurIPS BibTeX record has no page field, so no disputed third-party pagination was added. | [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2018/hash/1dc3a89d0d440ba31729b0ba74b93a33-Abstract.html) |
| `iyer2022activedendrites` | Corrected the first author from Rishabh Iyer to Abhiram Iyer and replaced `and others` with the complete six-author list. | [Frontiers in Neurorobotics](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.846219/full) |
| `chavlis2025dendrites` | Added article number 943. | [Nature Communications DOI record](https://api.crossref.org/works/10.1038/s41467-025-56297-9) |
| `lv2025dendritic` | Corrected the first author from Jinpeng Lv to Changze Lv; restored the full ten-author list and full title; added PMLR volume 267, pages 41682--41700, publisher, and official URL; removed the arXiv-only note because the paper has a proceedings version. | [PMLR](https://proceedings.mlr.press/v267/lv25c.html) |
| `francioni2026vectorized` | Replaced `and others` with the six authors and added Nature volume 652 and pages 1254--1263. | [Nature](https://www.nature.com/articles/s41586-026-10190-7) |

## Closest prior work added

| Key | Verified record | Authoritative source |
|---|---|---|
| `rossbroich2023disinhibitory` | Rossbroich and Zenke, NeurIPS 36, 64059--64082 (2023), DOI 10.52202/075280-2799. | [NeurIPS proceedings](https://papers.nips.cc/paper/2023/hash/ca22641c182b3b9608634edb4d09bc33-Abstract-Conference.html) |
| `galloni2026cellular` | Galloni, Peddada, Chennawar and Milstein, *Cell Reports* 45(4), 117159 (2026). | [Crossref DOI record](https://api.crossref.org/works/10.1016/j.celrep.2026.117159) |
| `greedy2026celltype` | Greedy et al., *bioRxiv* (2026), DOI 10.64898/2026.06.16.732595. | [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.06.16.732595v1) |
| `weis2025morphology` | Weis et al., *Nature Communications* 16, 3361 (2025); complete 48-author list retained. | [Nature Communications DOI record](https://api.crossref.org/works/10.1038/s41467-025-58763-w) |
| `kornfeld2020anatomical` | Kornfeld et al., *bioRxiv*, DOI 10.1101/2020.02.18.954354. The DOI publication year is 2020; a note records that version 2 was posted on 27 October 2025. | [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.02.18.954354v2); [PubMed version record](https://pubmed.ncbi.nlm.nih.gov/41279705/) |
| `bicknell2021synaptic` | Bicknell and Haeusser, *Neuron* 109(24), 4001--4017.e10 (2021). | [PubMed](https://pubmed.ncbi.nlm.nih.gov/34715026/); [Crossref DOI record](https://api.crossref.org/works/10.1016/j.neuron.2021.09.044) |
| `jordan2024conductance` | Jordan et al., *PLOS Computational Biology* 20(6), e1012047 (2024). | [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012047) |
| `rall1962dendrites` | Rall, *Annals of the New York Academy of Sciences* 96(4), 1071--1092 (1962), DOI 10.1111/j.1749-6632.1962.tb54120.x. | [Wiley publisher record](https://nyaspubs.onlinelibrary.wiley.com/doi/10.1111/j.1749-6632.1962.tb54120.x) |
| `gidon2012inhibition` | Gidon and Segev, *Neuron* 75(2), 330--341 (2012), DOI 10.1016/j.neuron.2012.05.015. | [PubMed](https://pubmed.ncbi.nlm.nih.gov/22841317/) |
| `ding2025wiring` | Ding, Fahey, Papadopoulos et al., *Nature* 640, 459--469 (2025), DOI 10.1038/s41586-025-08840-3. | [Nature publisher record](https://www.nature.com/articles/s41586-025-08840-3) |

`lv2025dendritic` was already present and therefore was corrected rather than added.

## Author preprints and disclosure assessment

| Key | Verified arXiv metadata | Relationship to this journal manuscript |
|---|---|---|
| `safaai2026localcredit` | Houman Safaai, Maceo Richards and Bernardo L. Sabatini, *Shunting Inhibition and Dendritic Branching Shape Local Credit Assignment*, arXiv:2607.03556, first posted 3 July 2026; current metadata is v2, updated 27 July 2026, primary class q-bio.NC. [arXiv record](https://arxiv.org/abs/2607.03556) | This is a direct predecessor of the journal manuscript's analytical factorization, local-learning experiments and conductance/shunting analyses. It must be cited in the manuscript and disclosed in the cover letter/submission as the relevant preprint. The journal paper should state the substantive extension: reconstructed MICrONS morphologies, sparse ancestry-defined routing, focal perturbations, task-alignment boundary, and any new journal-only analyses. |
| `safaai2026gainload` | Houman Safaai, Maceo Richards, Naeem Khoshnevis and Bernardo L. Sabatini, *When Branch-Local Shunting Helps: A Gain-Load-Alignment Principle for Dendritic E/I Networks*, arXiv:2607.24990v1, posted 27 July 2026, primary class q-bio.NC. [arXiv record](https://arxiv.org/abs/2607.24990) | This is not an earlier version of the credit-routing journal study. Its main question is forward population readout under additive versus shunting integration, rather than local credit transport through reconstructed trees. It is nevertheless a closely related concurrent manuscript by overlapping authors using the DendriNet framework and overlapping shunting/morphology concepts. It should be cited where the journal manuscript discusses when shunting improves computation, and disclosed as a related manuscript if the submission system or editor asks. The cover letter should distinguish its forward-computation results from the present paper's credit-assignment and anatomical-routing results. |

Neither arXiv record should be presented as independent external confirmation. The first is the paper's direct preprint lineage; the second is related work from the same research program.

## Records verified without substantive metadata changes

| Keys | Primary or registered source |
|---|---|
| `poirazi2003pyramidal` | [DOI/Crossref](https://api.crossref.org/works/10.1016/S0896-6273(03)00149-1) |
| `spruston2008pyramidal` | [DOI/Crossref](https://api.crossref.org/works/10.1038/nrn2286) |
| `larkum2013cellular` | [DOI/Crossref](https://api.crossref.org/works/10.1016/j.tins.2012.11.006) |
| `urbanczik2014dendritic` | [DOI/Crossref](https://api.crossref.org/works/10.1016/j.neuron.2013.11.030) |
| `carandini2012normalization` | [Nature Reviews Neuroscience](https://www.nature.com/articles/nrn3136). The article appeared online in 2011 but belongs to volume 13 (2012), so 2012 is retained. |
| `holt1997shunting` | [DOI/Crossref](https://api.crossref.org/works/10.1162/neco.1997.9.5.1001) |
| `bloss2016structured` | [DOI/Crossref](https://api.crossref.org/works/10.1016/j.neuron.2016.01.029) |
| `vogels2011inhibitory` | [DOI/Crossref](https://api.crossref.org/works/10.1126/science.1211095) |
| `fremaux2016threefactor` | [DOI/Crossref](https://api.crossref.org/works/10.3389/fncir.2015.00085) |
| `rumelhart1986learning` | [DOI/Crossref](https://api.crossref.org/works/10.1038/323533a0) |
| `nokland2016dfa` | [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2016/hash/d490d7b4576290fa60eb31b5fc917ad1-Abstract.html) |
| `guerguiev2017segregated` | [eLife DOI record](https://api.crossref.org/works/10.7554/eLife.22901) |
| `richards2019dendritic` | [DOI/Crossref](https://api.crossref.org/works/10.1016/j.conb.2018.08.003) |
| `whittington2019theories` | [DOI/Crossref](https://api.crossref.org/works/10.1016/j.tics.2018.12.005) |
| `lillicrap2020backprop` | [DOI/Crossref](https://api.crossref.org/works/10.1038/s41583-020-0277-3) |
| `payeur2021burst` | [DOI/Crossref](https://api.crossref.org/works/10.1038/s41593-021-00857-x) |
| `song2024prospective` | [DOI/Crossref](https://api.crossref.org/works/10.1038/s41593-023-01514-1) |
| `microns2025` | [Nature DOI record](https://api.crossref.org/works/10.1038/s41586-025-08790-w) |
| `cichon2015branch` | [DOI/Crossref](https://api.crossref.org/works/10.1038/nature14251) |

## Version-sensitive decisions

- The Kornfeld record uses 2020 because this is the year registered for the DOI and the year by which the work is indexed. The current 2025 revision is stated explicitly in `note`; silently changing the citation year to 2025 would break matching with Crossref and many bibliographic databases.
- The Koch book retains 1999 despite Oxford's 12 November 1998 online-publication field because the printed edition is copyrighted and conventionally cited as 1999.
- The Carandini and Heeger paper retains 2012 because its volume and issue are dated 2012, despite advance online publication in November 2011.
