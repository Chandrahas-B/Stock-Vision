# Stock-Vision
Stock market forecasting by taking images containing graphs of the stocks as input.

## Adaptibility and Impact:
  Traditional forecasting techniques are heavily dependent on tabular data for predicting the stock movement. Hence, our team wanted to try a unique approach to solve the forecasting problems. In this project, we extract graphs from images using various image processing techniques and use them to forecast the next 100-200 future values of that particular stock on the input image. This helps the user to decide if he/she has to invest in the stock.

## Code:
  The forecasting model was developed using the AI toolkit which provided optimizations to build deep learning models. After experimenting with many models and architecture, the best model was found which was a combination of Conv1d and LSTM layers. These layers were optimized by the toolkit which improved the speed of the training by almost 1.5x times with minimal loss in the accuracy.
  The dataset consists of graph images which were hand-picked by searching for the stocks and takiyyng screenshots of them. Each of the images contained a graph which was converted into a tabular data by performing bit-masking and other pre-processing techniques. The dataset comprised around 224 images which contains continuous graphs. These graphs were broken down into individual points and then used as the training set. Hence, the training set had a total of around 1.2 million unique records.
  
  
## Design:
  After trying out multiple moedels and different architectures, the model was found to perform well when it had a combination of multiple Conv1D and LSTM layers without any skip connections. Using this design, the model was able to achieve a good Mean Squared Error score.
  ![image](https://user-images.githubusercontent.com/84665480/225654048-566e8770-8884-4b4a-b067-f9415a91b233.png)
The model was trained with 150k parameters and also consists of around 1k non-trainable parameters which were assigned in Layer and Batch Normalization.
Hence, it is capable of generating accurate results in short duration of time and is highly scalable as it is trained on around 60-70 trending stocks of 2022-23.

## Usability:
  The project was aimed to be used on a daily basis which can assist users in understand the graph movement. This can also interest new users to invest in stocks and help them understand the graph movements which can help them in making wise decisions in investing.
  The project has covered functionalities such as image processing for extracting graphs and can forecast around 150 values. 
  
  ## Prototype build:
    The project has been implemented with various functionalities which can be used as a viable product to forecast stocks on a daily basis. 
    This project can be used as a baseline prototype to create a more robust product.
