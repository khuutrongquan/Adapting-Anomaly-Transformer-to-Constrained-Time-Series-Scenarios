# Adapting Anomaly Transformer to Constrained Time Series Scenarios

This repository contains the implementation and experimental results of our research on Anomaly Transformer model to constrained time series scenarios.

## Abstract
Unsupervised time series anomaly detection plays a pivotal role in real-world applications where annotated data is scarce or unavailable. The primary challenge lies in constructing effective anomaly scoring mechanisms capable of distinguishing subtle abnormal patterns from normal behavior. While the Anomaly Transformer has achieved state-of-the-art results on long time series, our empirical analysis indicates a marked performance degradation when it is applied to constrained or low-dimensional datasets. We attribute this limitation to the model’s dependence on a 1D-CNN feature extractor, which often fails to capture rich temporal and dynamic dependencies—thereby diminishing the utility of the subsequent self-attention mechanism. To overcome this shortcoming, we introduce a hybrid feature extraction framework that integrates Temporal Convolutional Networks (TCNs) and standard CNNs to capture both global and local temporal features more effectively. By replacing the original embedding layer with our hybrid encoder, the model gains a more expressive representation of time-dependent patterns, improving its anomaly detection capability without additional supervision. Experimental results on three benchmark low-dimensional datasets—UCR, ECG, and 2D-Gesture—demonstrate that the proposed model consistently outperforms the baseline Anomaly Transformer in terms of F1 score, validating its robustness and generalization in constrained settings.

## Main Contributions

1. **Overall Architecture**
   ![Proposed Anomaly Transformer model](Images/ProposedAnomalyTransformer.jpg)
   - Proposed a Hybrid Encoder to improve an ability of capturing temporal and dynamic dependencies and reduce over-reliance on self-attention mechanism of Transformer.
   - Achieve F1 score improvements of approximately 18% on the UCR dataset, 1.2% on ECG, and 0.5% on the 2D-Gesture dataset.
2. **Proposed Hybrid Encoder**
   ![Hybrid Encoder](Images/HybridEncoder.jpg)
   - Integrate Temporal Convolutional Networks (TCNs) and Convolutional Neural Networks (1D-CNNs)
   - 
3. **Training and Testing Strategy**

## Authors

- Khuu Trong Quan¹ (khuutrongquan220405@gmail.com)
- Huynh Cong Viet Ngu²* (nguhcv@fe.edu.vn)

^{1}Department of Software Engineering, FPT University, Ho Chi Minh, Vietnam
^{2}Department of Computing Fundamental, FPT University, Ho Chi Minh, Vietnam
\* Corresponding author

