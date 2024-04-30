# Stochastic Activity and Data Aware Process Models

Code for the chapter "*Stochastic Activity and Data Aware Process Models*" of the thesis "*Machine Learning for Probabilistic and Attribute-Aware Process Mining*".

Required packages are in the *requirements.txt* file. To install with pip:

```
pip install requirements.txt
```

Before using the script, unzip the "*logs.zip*" directory.
To run the algorithm on the running example proposed in the chapter:

```
python main.py running-example --compute True
```

To evaluate the method on the real-world event logs:

```
python main.py method-evaluation --compute True
```

Results are in the directory "*results*" and if the parameter "*compute*" is not specified, the plotted results are the onse contained in that directory.
