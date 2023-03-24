# 30 days of ML

## Plan: implement 5 canonical papers that are foundations of modern DL 

### Paper 1: Backprop by Hinton (March 20)

Concept: Backprop & Autograd\
Link: https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf \
Implementation: MLP trained on MNIST dataset using backprop & vanilla grad descent

### Paper 2: Deep Residual Learning for Image Recognition (March 27)

Concept: Residual Networks + Convolution\
Link: https://arxiv.org/abs/1512.03385 \ 
Implementation: Train a ResNet on the CIFAR-10 dataset

#### Bonus: 
Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al. (2015) \
Link: https://arxiv.org/abs/1505.04597
Implementation: Train over publicly available dataset like the ISBI 2012 segmentation challenge dataset 

### Paper 3: Attention is All You Need (April 4)

Concept: Attention\
Link:https://arxiv.org/abs/1706.03762 \
Implementation: train a transformer model on a text classification or translation task using a smaller dataset, such as the IWSLT dataset.

### Paper 4: Generative Adversarial Networks (April 11)

Concept: Discriminators\
Link: https://arxiv.org/abs/1406.2661 \
Implementation: train a simple GAN on the MNIST or CIFAR-10 dataset.

### Paper 5: Deep Q-Netowrk (April 18)

Concept: Reinforcement Learning \
Link: https://arxiv.org/abs/1312.5602 \
Implementation: Use OpenAI Gym environments such as CartPole, LunarLander, or Atari games.

### NOTES

Suggestions by GPT-4:

Here are five additional influential papers that cover a wide range of deep learning topics, along with the datasets and benchmarks for each implementation:

| Paper | Link | Why | Dataset | Benchmark |
|-------|------|-----|---------|-----------|
| Long Short-Term Memory | https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735 | LSTMs are a fundamental building block of many sequential models and have been widely used in various NLP, time series, and speech recognition tasks. | IMDB sentiment classification dataset or Penn Treebank (PTB) dataset for language modeling. | Measure the classification accuracy for the IMDB dataset, or the perplexity for the PTB dataset. |
| Dropout: A Simple Way to Prevent Neural Networks from Overfitting | http://jmlr.org/papers/v15/srivastava14a.html | Dropout is a simple but powerful regularization technique that helps prevent overfitting and is used in various deep learning models. | MNIST, CIFAR-10, or another dataset where overfitting might be an issue. | Compare the test set performance (accuracy) of a model with and without dropout to demonstrate the effectiveness of dropout as a regularization technique. |
| Deep Q-Networks | https://www.nature.com/articles/nature14236 | DQN is a foundational work that combines deep learning with reinforcement learning and is responsible for the rapid development of deep reinforcement learning. | Use OpenAI Gym environments such as CartPole, LunarLander, or Atari games. | Measure the average cumulative reward over multiple episodes to assess the performance of the DQN agent. |
| U-Net: Convolutional Networks for Biomedical Image Segmentation | https://arxiv.org/abs/1505.04597 | U-Net is a widely-used architecture for image segmentation tasks, particularly in medical imaging. It has inspired many other architectures for segmentation tasks. | Use a publicly available dataset like the ISBI 2012 segmentation challenge dataset (http://brainiac2.mit.edu/isbi_challenge/home) or another biomedical image segmentation dataset. | Measure the Intersection over Union (IoU) or the Dice coefficient to assess the performance of the U-Net model. |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | https://arxiv.org/abs/1810.04805 | BERT revolutionized the NLP field by introducing a pre-trained transformer model that achieved state-of-the-art performance on numerous NLP tasks with fine-tuning. | Use the GLUE benchmark dataset (https://gluebenchmark.com/) or a smaller text classification dataset like SST-2 (Stanford Sentiment Treebank) or CoLA (Corpus of Linguistic Acceptability). | Measure the performance of the fine-tuned BERT model in terms of accuracy, F1 score, or other relevant metrics depending on the chosen dataset and task. |
| ImageNet Classification with Deep Convolutional Neural Networks | http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf | This paper introduced the AlexNet architecture which achieved a significant improvement in image classification on the ImageNet dataset and paved the way for the deep learning revolution in computer vision. | ImageNet dataset | Measure the top-1 and top-5 classification accuracy on the ImageNet validation set. |
| Playing Atari with Deep Reinforcement Learning | https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf | This paper introduced the DQN algorithm which achieved human-level performance on several Atari games using only pixel inputs and reinforcement learning. | Atari games | Measure the average cumulative reward over multiple episodes to assess the performance of the DQN agent on various Atari games. |
| EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | https://arxiv.org/abs/1905.11946 | This paper proposed an efficient model scaling method that achieves state-of-the-art accuracy on both image classification and object detection tasks while reducing the number of parameters and FLOPs. EfficientNet combines compound scaling, a novel uniform scaling method, and a new architecture search algorithm | 
| YOLOv3: An Incremental Improvement | https://arxiv.org/abs/1804.02767 | This paper introduced YOLOv3, a state-of-the-art object detection model that achieves real-time performance on standard GPUs. YOLOv3 introduced several improvements over its predecessors, including feature extraction, multiscale prediction, and better anchor box selection. | COCO dataset or another object detection dataset | Measure the mean average precision (mAP) at different intersection over union (IoU) thresholds to assess the performance of the YOLOv3 model. |
| Mask R-CNN | https://arxiv.org/abs/1703.06870 | This paper introduced Mask R-CNN, a state-of-the-art object detection and instance segmentation model that extends the Faster R-CNN architecture with a branch for predicting object masks. Mask R-CNN achieved state-of-the-art performance on the COCO dataset. | COCO dataset or another object detection and instance segmentation dataset | Measure the mean average precision (mAP) at different IoU thresholds and the segmentation quality using the mean average precision for instance segmentation (AP segm) to assess the performance of the Mask R-CNN model. |



