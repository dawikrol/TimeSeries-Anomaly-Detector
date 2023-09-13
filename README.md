# TimeSeries Anomaly Detector

The repository contains the source code for the implementation of my master's thesis on the detection and prediction of
anomalies in multidimensional time series data.

## Table of Contents
- [Master thesis summary](#master-thesis-summary)
- [Presentation of the problem](#presentation-of-the-problem)
- [Results](#results)
- [Code and implementation notes](#code-and-implementation-notes)

---

### Master thesis summary
In today's rapidly advancing technological world, the analysis and prediction of time series data are integral
components of many fields, including industry, particularly in the context of the growing role of the Internet of
Things (IoT). This thesis conducts a comprehensive analysis of multidimensional time series data with a focus on
predicting and detecting anomalies during the electrical discharge machining (EDM) process. Within this study, a review
of existing solutions was conducted, emphasizing the utilization of machine learning techniques. Specifically, two
primary models were proposed and implemented: a supervised Long Short-Term Memory (LSTM) neural network model and an
unsupervised Convolutional Neural Network (CNN)-based autoencoder model. Both models demonstrated effectiveness in
predicting time series behavior, including prediction and anomaly detection. The autoencoder model proved
particularly valuable, enabling efficient labeling of unlabeled data, which is crucial for further data analysis.

### Presentation of the problem
During the process of electrical discharge machining, there can be instances of material discontinuity, which is an
undesirable phenomenon. When this occurs in time series data representing the characteristics of parameter values
related to the process, anomalies become evident. These anomalies are indicated by a binary machine parameter that
signals the occurrence of a breakthrough. This parameter has a delay (Fig 1).

![image](https://github.com/dawikrol/TimeSeries_Anomaly_Detector/assets/63808220/95423171-413b-4686-99e8-9145c4f199d2)


_Fig 1: Examples of anomalies. On the left there is a sample with a break, on the right there is a sample without a
break. BT-Detect - binary parameter indicating breakdown detection._

The aim of the thesis was to present a preliminary solution that demonstrates how the use of machine-learning tools can
advance the marking of the BT-Detect parameter and predict faults even before they occur.

### Results
LSTM and Autoencoder models were implemented. Implementation details and the decisions made during this process are
described in the thesis. When it becomes available, a link will be provided here.

Both investigated models demonstrate effectiveness in the context of early prediction of the BT-Detected parameter,
which serves as an anomaly detection indicator (Figure 2). However, a direct comparison of the performance of these two
approaches is not straightforward. The LSTM-based model directly forecasts the target parameter, while the
autoencoder-based model takes advantage of the fact that this parameter has a delay compared to the occurrence of
anomalies, which is its characteristic feature. Thus, the autoencoder-based model eliminates the delay in the target
parameter, providing a significant advantage in early anomaly detection. It's worth noting that these two models don't
necessarily have to compete with each other but can be used synergistically.

![image](https://github.com/dawikrol/TimeSeries_Anomaly_Detector/assets/63808220/96483645-bf22-4495-bbe4-ba9f6b3f7c01)

_Fig 2: Anomaly detection by LSTM model and Autoencoder._

The most significant challenge encountered during the study was the limited availability of labeled data. In the context
of having a highly effective anomaly detection model like the autoencoder model, there is an opportunity to effectively
label data that previously lacked clear labels indicating anomalies. As a result, this process can significantly
increase the number of samples in the dataset. Furthermore, such an approach allows for acquiring additional data that
can be labeled with much greater precision than the original BT-Detect parameter. This is a crucial aspect contributing
to the improvement of model quality and precision.

The collaborative use of these two approaches can lead to even better results in the realm of early anomaly detection.

### Code and implementation notes
The code is not a production version. It was used for conducting experiments and fine-tuning the model based on the
results, and it was utilized during the implementation of the master's thesis. Therefore, the implementation deviates
from the optimal version (e.g., most loops could be simplified and merged). Unfortunately, the data used for training
the model as well as the trained model itself are legally protected and cannot be shared.

It is possible that certain improvements will be added in the future:

- A CLI (Command Line Interface) layer to facilitate easy execution of the model and the ability to input custom
  hyperparameters.
- Dummy data for training and validating the model.
