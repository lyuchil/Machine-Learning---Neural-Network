# Project Goatfish Readme

### Riley Blair, Stephen Fanning, Yu-Chi Liang, William Ryan

[Goatfish Frontend](http://goatfish.s3.us-east-2.amazonaws.com/index.html)
[Goatfish Neural Network Repository](https://github.com/lyuchil/Machine-Learning---Neural-Network)
[Goatfish Frontend Repository](https://github.com/lyuchil/Machine-Learning---Frontend)


### Important note: Please make sure you access the site through *HTTP* and NOT *HTTPS* to ensure the webapp is able to communicate with the model

## Key Takeaways and Questions

- When troubleshooting a model during development, a useful strategy we discovered is attempting to force the model to overfit and memorize just one specific datapoint. Early on this was helpful as it allowed us to test model designs on their very base learning capabilties before training on the full dataset.
- Time complexity is crucial when training a model. When training with high amounts of data, greater time complexities become costly in both training and evaluating a model.
- This network is remarkably effective for this kind of pattern recognition, espeically with the already complex and difficult nature of chess.


## Neural Network Information

For our training we utilized WPI's Turing Cluster. All of the shell scripts we used to run on the cluster are in the repository for the network (including how we downloaded and parsed the data from lichess, and how we train the model ). We trained off of data from Lichess's database, specifically on games ocurring in October 2023, December 2023, Feburary 2024, March 2024 and reserved November 2023 and January 2024 for evaluation and testing respectively. We also have a requirements.txt that can be used with pip to get requirements into the environment.

Link to lichess database: [Lichess Database](https://database.lichess.org/)

## Frontend Information

Our frontend is built with Vite + React, chess.js, and react-chessboard. You should be able to clone the repository, enter the "chess_front_end" directory, run `npm install`, then run `npm run dev` to get started.
