# Commit message classification

This repository contains code for training commit message classifier and some examples of usage.

## Structure

* datasets
  * dataset1 - this is a dataset divided into 3 parts (train - 70%, val - 15%, test - 15%), marked up by the authors of the article   [ Multi-label Classification of Commit Messages Using Transfer Learning](https://www.researchgate.net/profile/Mohamed-Wiem-Mkaouer-2/publication/348228961_Multi-label_Classification_of_Commit_Messages_using_Transfer_Learning/links/61eacfc2c5e3103375ae596d/Multi-label-Classification-of-Commit-Messages-using-Transfer-Learning.pdf). It is used to train and evaluate the model.
  * dataset2 - these are the commit messages from the test sample of the [NNGen](https://github.com/Tbabm/nngen) dataset.
* src
  * analyze
    * analyze_dataset1 - getting the value of the metrics on the test dataset to check the effectiveness of the model.
    * analyze_dataset2 - label prediction with a trained model to analyze results.
  * metrics - computing of 8 metrics for our model.
  * predict - helper function for label prediction.
  * train - main module. Contains the model itself and the code for training it.

## Usage

1 **Install**
```
git clone https://github.com/al-volkov/commit-message-classification.git
pip install -r requirements.txt
```
2 **Configure**
    
You should set the parameters you need in [`src/analyze/analyze.yaml`](src/analyze/analyze.yaml) and [`src/train/train_config.yaml`](src/train/train_config.yaml)

3 **Train**

Now you need to run [`src/train/train.py`](src/train/train.py)

4 **Test**

Once the model has trained, you can test it: [`src/analyze/analyze_dataset1.py`](src/analyze/analyze_dataset1.py)

5 **Analyze dataset2**

If you think the predictions are accurate enough, you can get results for dataset2: [`src/analyze/analyze_dataset2.py`](src/analyze/analyze_dataset2.py)

##Metrics

Evaluated on [`datasets/dataset1/test.csv`](datasets/dataset1/test.csv)

| metric | accuracy | f1_score_micro | f1_score_macro | precision_micro | precision_macro | recall_micro | recall_macro | hamming_loss |
|--------|----------|----------------|----------------|-----------------|-----------------|--------------|--------------|--------------|
| value  | 0.73     | 0.84           | 0.85           | 0.89            | 0.89            | 0.8          | 0.81         | 0.12         |

## Dataset2 analysis
![a1](https://user-images.githubusercontent.com/70965603/198382352-7581b225-3553-4a57-a4d2-d8f33ffa2a65.png)
![a2](https://user-images.githubusercontent.com/70965603/198382361-d928f475-cec1-496b-9349-e0f7f8bf01bd.png)
![a3](https://user-images.githubusercontent.com/70965603/198382365-afa4a95c-935a-4c88-a07e-880356f77885.png)



