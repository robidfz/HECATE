
# HECATE

This repository is the official code for the paper "Towards Hepatic Cancer Detection with Bayesian Networks for Patients Digital Twins Modelling" by Roberta De Fazio, Adrian Bartos, Viviana Leonetti, Stefano Marrone and Laura Verde.

![Workflow](https://github.com/robidfz/HECATE/blob/main/Figures/LDTH-workflow.png)

## Citation
Please cite our work if you find it useful for your research and work.

```
@article{DeFazio2024,
  title = {Towards Hepatic Cancer Detection with Bayesian Networks for Patients Digital Twins Modelling},
  volume = {246},
  ISSN = {1877-0509},
  DOI = {10.1016/j.procs.2024.09.598},
  journal = {Procedia Computer Science},
  publisher = {Elsevier BV},
  author = {De Fazio,  Roberta and Bartoş,  Adrian and Leonetti,  Viviana and Marrone,  Stefano and Verde,  Laura},
  year = {2024},
  pages = {5104–5113}
}
```

## Dependencies

The code relies on the following Python 3.9 + libs.

Packages needed are:
* numpy
* pandas
* matplotlib
* pgmpy 0.1.25
* networkx 3.3


## Data
The data are not available due to privacy policies.
Here we provide an example of data structure with non real instances.

| Age | Varsta | MRI | PET-CT | CT | Tumor Type | I PRE   | II PRE | III PRE| IV PRE | V PRE | VI PRE  | VII PRE | VIII PRE | I POST | II POST | III POST | IV POST | V POST  | VI POST | VII POST | VIII POST |
|-----|--------|-----|--------|----|------------|---------|--------|--------|--------|-------|---------|---------|----------|--------|---------|----------|---------|---------|---------|----------|-----------|
| 68  | F      | 0   | 1      | 0  | 0          | 0       | 1      | 0      | 1      | 0     | 1       | 0       | 1        | 0      | 1       | 1        | 0       | 0       | 1       | 1        | 1         | 
| 56  | M      | 1   | 1      | 1  | 2          | 0       | 1      | 1      | 1      | 0     | 1       | 0       | 1        | 1      | 1       | 1        | 0       | 0       | 1       | 1        | 1         |
| 81  | M      | 0   | 0      | 1  | 1          | 1       | 1      | 0      | 1      | 0     | 0       | 0       | 1        | 1      | 1       | 0        | 1       | 0       | 1       | 1        | 1         | 


## How to use

To run the code you have to provide the name of the configuration file and the dataset path:

```
python 00_BN.py config.ini *datasetpath*
```


