# An Lstm-based Distributed Scheme For Data Transmission Reduction Of Iot Systems [(Go-to-Paper)](https://www.sciencedirect.com/science/article/abs/pii/S0925231221016295?via%3Dihub&fbclid=IwAR3bNM8A6BXq4zaR1quTLNrsUXTaVSlzOaM8AVkWRC51be1GfIDy6Zg8m8Q)

## Abstract
The growth of the number of connected devices in Internet of Things (IoT) systems causes a huge increase in network traffic. Thus, there is a significant demand for systems that can predict the measurements of the distributed IoT-based applications to mitigate the increasing network traffic. Existing methods utilized a distributed scheme, i.e., dual prediction schemes (DPS), to achieve this task. The idea of this scheme is based on deploying a predictive model on the data sources (i.e., sensors) and the fusion center in a distributive manner. The state-of-the-art results can be simply reached using predictive models, e.g., adaptive filters. Deep learning-based approaches are not well utilized to address this problem. In this context, we proposed a distributed scheme utilizing fog and edge computing technologies. The proposed DPS includes an LSTM predictive model to reduce the data transmission instances of connected IoT devices; it is called LSTM-DPS. In addition, we proposed an updating mechanism that updates the LSTM model according to a set of tracking parameters that observes the model behavior during the deployment due to the changes of data properties through time. The proposed updating mechanism guarantees that the deployed LSTM model is identical at the data source and fusion center. The LSTM-DPS is evaluated using two real datasets. The obtained results show that the LSTM-DPS outperforms state-of-the-art methods in terms of communication reduction ratios.



## Citing

If you use the proposed simulation in your work, please cite the accompanying [paper]:

```bibtex
@article{FATHALLA2021,
title = {An LSTM-based Distributed Scheme for Data Transmission Reduction of IoT Systems},
journal = {Neurocomputing},
year = {2021},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2021.02.105},
url = {https://www.sciencedirect.com/science/article/pii/S0925231221016295},
author = {Ahmed Fathalla and Kenli Li and Ahmad Salah and Marwa F. Mohamed},
keywords = {Deep learning, distributed scheme, dual prediction, edge computing, IoT, LSTM},
abstract = {The growth of the number of connected devices in Internet of Things (IoT) systems causes a huge increase in network traffic. Thus, there is a significant demand for systems that can predict the measurements of the distributed IoT-based applications to mitigate the increasing network traffic. Existing methods utilized a distributed scheme, i.e., dual prediction schemes (DPS), to achieve this task. The idea of this scheme is based on deploying a predictive model on the data sources (i.e., sensors) and the fusion center in a distributive manner. The state-of-the-art results can be simply reached using predictive models, e.g., adaptive filters. Deep learning-based approaches are not well utilized to address this problem. In this context, we proposed a distributed scheme utilizing fog and edge computing technologies. The proposed DPS includes an LSTM predictive model to reduce the data transmission instances of connected IoT devices; it is called LSTM-DPS. In addition, we proposed an updating mechanism that updates the LSTM model according to a set of tracking parameters that observes the model behavior during the deployment due to the changes of data properties through time. The proposed updating mechanism guarantees that the deployed LSTM model is identical at the data source and fusion center. The LSTM-DPS is evaluated using two real datasets. The obtained results show that the LSTM-DPS outperforms state-of-the-art methods in terms of communication reduction ratios.}
}
```
[paper]: https://www.mdpi.com/1424-8220/21/17/5777
