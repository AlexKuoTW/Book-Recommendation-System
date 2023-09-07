# Book-Recommendation-System
### 1. Book-Crossing Dataset
This is a dataset collected from a book crossing (圖書漂流) community, containing 278,858 users with 1,149,780 ratings about 271,379 books.
There are 3 csv files in the folder above: Ratings.csv contains 3 columns User-ID, ISBN and Book-Rating.
This file is the training data, containing 1,034,802 (90%) records, by random and stratified sampling. Users.csv contains the detailed information for all 278,858 users, including their locations and ages. Books.csv contains the titles, authors, and much more detailed information about all 271,379 books. Further descriptions can be found in the original dataset website here.
Finally, for those students using this dataset, your recommendation system would be tested by the 114,978 records preserved, and ranked by the accuracy to compare with other students. ±1 from the ground truth rating would still be considered accurate. 
