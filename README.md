# credit_default_model
Built credit card default prediction model with fine-tuned hyperparameters using GridSearchCV and optimized time complexity with Google Colab

## Inspiration
Throughout my past co-ops I worked with data in all sorts of applications from cancer research to marketing analytics to zero-emission vehicles. However, as an avid investor and personal finance enthusiast, I have always wanted to gear my career interest toward finance and the endless applications of data involved in that industry. **So I used this chance to practice working with financial data while continuing to learn new concepts in machine learning and data analytics**.

## Dataset
I used the **Taiwanese credit default dataset** from Kaggle, which contains **30,000 rows** and 25 parameters. It contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005

## How I built it
If you look at the Jupyter Notebook, I divided and wrote it down into 5 steps. But briefly speaking, I evaluated and compared 8 different ML models, by **cross-validating** them and averaging out the accuracies to pick the best ones. Then, I used **GridSearchCV** and **RandomizedSearchCV** to **fine-tune the hyperparameter** to get the best-performing models with the best-performing combination of parameters. In the end, although the ones that I fine-tuned ended up being similar, **XGBoost** turned out to be the highest-performing model.

## Challenges I ran into and what I gained from it
**Runtime matters**. Initially, while tuning the hyperparameters with grid search, I realized that it took an **extremely** long time. So I had to start researching how to speed up the runtime. I discovered two main methods to cut down runtime drastically. When I first learned about GridSearchCV, it was a pretty straightforward concept to me because all it was doing was exhaustively testing every possible combination and running it. I quickly came to realize how time-consuming it is and then learned about **RandomizedSearchCV** and how it uses a probability distribution to optimally take samples of the hyperparameters, which cut down runtime drastically. Second, I learned about **using additional GPU and TPU resources** using Google Colaboratory. Since it also used a similar notebook format to Jupyter, I found it quite straightforward to work with it.
