from cgi import print_arguments
from itertools import count
from statistics import mean, mode
from unicodedata import name, normalize
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from surprise import Prediction, Reader, Dataset
from surprise import SVD, NMF, accuracy
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from math import sqrt
from numpy import isin, nan as na, size
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from collections import defaultdict
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import pickle


# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# Question/Idea 
#1. It might be more useful for our model if we simplified this to give each book a unique identifier
#2. There isn't really anything we can do about that, but we should really remove them from the dataset 
#   as we won't be able to access the title of the book to make a recommendation even if the model can use them.

def main():
#step1. import data to df
    warnings.filterwarnings("ignore") 
    Books_path = r'.\\Project1\\Book-Crossing Dataset\\Books.csv'
    Ratings_path = r'.\\Project1\\Book-Crossing Dataset\\Ratings.csv'
    Users_path = r'.\\Project1\\Book-Crossing Dataset\\Users.csv'

    Books_df = pd.read_csv(Books_path, encoding='latin-1', on_bad_lines='skip', sep=';')
    Ratings_df = pd.read_csv(Ratings_path, encoding='latin-1', on_bad_lines='skip', sep=';')
    Users_df = pd.read_csv(Users_path, encoding='latin-1', on_bad_lines='skip', sep=';')
    # pd.set_option('display.max_columns', None) #max_rows
    Books_df.drop(columns=["Image-URL-S","Image-URL-M","Image-URL-L"],axis=1, inplace=True)
    pd.set_option('max_colwidth', 20) #default 50
    # print(Books_df.iloc[:10,0:])
    # print(Ratings_df.iloc[:10,0:])
    # print(Users_df.iloc[:10,0:])
    # Books_df.to_csv(r'.\\Project1\\Book-Crossing Dataset\\After\\Books_after.csv')
    # Ratings_df.to_csv(r'.\\Project1\\Book-Crossing Dataset\\After\\Ratings_after.csv')
    # Users_df.to_csv(r'.\\Project1\\Book-Crossing Dataset\\After\\Users_after.csv')

#step2. merge data
    final_df = pd.merge(Books_df,Ratings_df,how='outer',on='ISBN')
    final_df = pd.merge(final_df,Users_df,how='outer',on='User-ID')
    # final_df.drop(final_df.columns[[0]],  axis=1, inplace=True)
    final_df.to_csv(r'.\\Project1\\Book-Crossing Dataset\\final_before.csv',index=False) #index=false去掉dataframe的第一欄

#step3. check any missing value
    # print(final_df.describe().round(2))
    # print(final_df.info())
    print(final_df.isnull().sum())
    # final_df.dropna(inplace=True)

#step4. start finding the messedup data 
    # for i in final_df:
    #     print({(i)},':',final_df[i].unique())
    # result = isinstance(final_df[i], str)

#step4-1. cleaning 'Year of public'
  #method1. isin(equal to removed)
    # final_df = final_df[~final_df['Year-Of-Publication'].isin(['DK Publishing Inc', 'Gallimard'])] #找出not isin key word的

  #method2. replace/loc/iloc
    final_df['Year-Of-Publication'] = final_df['Year-Of-Publication'].replace('DK Publishing Inc',2000)
    final_df['Publisher'] = final_df['Publisher'].replace('http://images.amazon.com/images/P/078946697X.01.THUMBZZZ.jpg',"DK Publishing Inc")
    final_df.loc[final_df['ISBN'] == '078946697X','Book-Author'] = "Michael Teitelbaum"
    final_df.loc[final_df['ISBN'] == '0789466953','Book-Author'] = "James Buckley"
    final_df['Year-Of-Publication'] = final_df['Year-Of-Publication'].replace('Gallimard',2003)
    final_df['Publisher'] = final_df['Publisher'].replace('http://images.amazon.com/images/P/2070426769.01.THUMBZZZ.jpg',"Gallimard")
    final_df.loc[final_df['ISBN'] == '2070426769','Book-Author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
    # print(final_df.loc[((final_df['Year-Of-Publication'] == 2000) & (final_df['Publisher'] == 'DK Publishing Inc')),:])
    final_df['Year-Of-Publication'] = pd.to_numeric(final_df['Year-Of-Publication']) # equal to df['Country'] = df.Country.astype('int32')
    # print(final_df['Year-Of-Publication'].sort_values().unique)
    final_df.loc[((final_df['Year-Of-Publication'] == 0) | (final_df['Year-Of-Publication'] > 2023))] =np.nan
    # print(df1['Year-Of-Publication'].unique()) #[0 2030 2038 2050 2026 2037 2024]
    # final_df = final_df[~final_df['Year-Of-Publication'].isin([0, 2030, 2038, 2050, 2026, 2037, 2024])] #Delete value is in list[]
    # final_df = final_df.reindex(columns=['ISBN','User-ID','Location','Age','Book-Title','Book-Author','Year-Of-Publication','Publisher','Book-Rating'])
    final_df['Year-Of-Publication'].fillna(final_df['Year-Of-Publication'].mean(), inplace=True)
    final_df['Year-Of-Publication'] = final_df['Year-Of-Publication'].astype(np.int64)

#step4-2. cleaning 'Age'
    final_df.loc[((final_df['Age'] == 0) | (final_df['Age'] > 105))] = np.nan
    # print(final_df['Age'].isnull().sum())
    final_df['Age'].fillna(final_df['Age'].mean(), inplace=True)
    # final_df['User-ID'].dropna(inplace=True)
    final_df['User-ID'] = final_df['User-ID'].astype('Int64')
    # final_df['Year-Of-Publication'] = final_df['Year-Of-Publication'].astype(np.int64)
    # final_df['Book-Rating'] = final_df['Book-Rating'].astype(np.int64)
    final_df['Age'] = final_df['Age'].astype(np.int64)
    # print(final_df['Age'].value_counts())

#step4-3. cleaning 'Location'
    location_c = final_df['Location'].str.split(',', n=2, expand=True)
    # print(location_c)
    final_df['City'] = location_c[0]
    final_df['State'] = location_c[1]
    final_df['Country'] = location_c[2]
    final_df.drop(columns=['Location'],inplace=True)
    final_df = final_df.reindex(columns=['ISBN','User-ID','Age','Country','State','City','Book-Title','Book-Author','Year-Of-Publication','Publisher','Book-Rating'])
    # print(final_df.shape)

#step5. visualable
#step5-1. visualable Users-Count
    # final_df.to_csv(r'.\\Project1\\Book-Crossing Dataset\\final_2.csv',index=False) #index=false去掉dataframe的第一欄
    pd.set_option('display.max_columns', None)
    pd.set_option("display.max_rows",30)
    top_users = final_df.groupby('User-ID').ISBN.count().sort_values(ascending = False)
    # print('Total\'s User have(Before remove 0) :',top_users.count()) # total user have 277,286
    # # print(final_df.loc[(final_df['User-ID'] == 198193),:]) # 可知道ISBN是NA的就為0
    # # print((top_users[top_users.values == 0].index)) # value==0的index是哪些ID
    # print('Value = 0\'s User have: ',top_users.where(top_users==0).count()) # value==0的總共有幾個User
    top_users[top_users.values == 0] =np.nan # drop掉這些value==0的User
    top_users.dropna(inplace = True)
    # print('Total\'s User have(After remove 0) :',top_users.count()) # total user have 277,286

    # print('Total\'s User over 1000 :',top_users.where(top_users>1000).count())
    # barplot(final_df,'User-ID','User-ID') #取前25個
    # histogram(top_users.where(top_users>1000)) #這代表user count小於50的有幾個
    # top_30 = top_users[:20]
    # print(top_30)

#step5-2. visualable Rating-Count
    # final_df = final_df[final_df['Book-Rating'] != 0]
    # histogram(final_df['Book-Rating'])
    # countplot(final_df)

    # ds = final_df['Book-Rating'].value_counts().to_frame().reset_index()
    # ds.columns = ['value', 'count']
    # ds=ds.drop([0])
    # fig = go.Figure(go.Bar(
    #     y=ds['value'],x=ds['count'],orientation="h",
    #     marker={'color': ds['count'], 
    #     'colorscale': 'sunsetdark'},  
    #     text=ds['count'],
    #     textposition = "outside",
    # ))
    # fig.update_layout(title_text='Rating Count',xaxis_title="Value",yaxis_title="Count",title_x=0.5)
    # fig.show()

#step5-3. visualable Books-Count (Q'ty not points)
    final_df.rename(columns={'Book-Rating' : 'Book_Rating'}, inplace=True)
    # top_books = final_df.groupby('Book-Title').ISBN.count().sort_values(ascending = False)
    top_books = final_df.groupby('Book-Title').Book_Rating.count().sort_values(ascending = False)
    # print('Total\'s Books have(Before) :',top_books.count())
    # print('Value = 0\'s Books have: ',top_books.where(top_books==0).count()) # value==0的總共有幾個User
    top_books[top_books.values == 0] =np.nan
    top_books.fillna(top_books.mean(), inplace=True)
    # print('Total\'s Books have(After) :',top_books.count()) # total user have 277,286
    # top_25_books = final_df['Book-Title'].value_counts()[:25]
    # print('Total\'s books over 200 :',top_books.where(top_books>200).count())
    # barplot_partial(top_25_books) # total books have 238,289
    # histogram(top_books.where(top_books>200))
    # histogram(top_books.where(top_books<10))

#step6. unique th3 ISBN
    final_df.dropna(subset=['User-ID'], inplace=True)
    final_df.dropna(subset=['Book-Title'], inplace=True)
    final_df[final_df.values == 0] =np.nan
    final_df.dropna(subset=['Book_Rating'], inplace=True)
    print(final_df.shape)
    # print(final_df.isnull().sum())
    # print(final_df[final_df['Book-Author'].isnull()].index.tolist())
    # print(final_df.iloc[597310,:])

#----------------------------------------unique isbn opened if need---------------------------------------
    # final_df.rename(columns={'Book-Title' : 'Book_Title'}, inplace=True)
    # multiple_isbns = final_df.groupby('Book_Title').ISBN.nunique() # Pandas nunique() 用于获取唯一值的统计次数
    # # print(multiple_isbns.value_counts())
    # more_mult_isbns = multiple_isbns.where(multiple_isbns>1)
    # more_mult_isbns.dropna(inplace=True) # remove NaNs, which in this case is books with a single ISBN number
    # # print(multiple_isbns.sort_values(ascending=False)[:5])
    # # print(len(more_mult_isbns))

    # # def make_isbn_dict(df):
    # #     title_isbn_dict = {}
    # #     for title in more_mult_isbns.index:
    # #         isbn_series = df.loc[df.Book_Title==title].ISBN.unique() # returns only the unique ISBNs
    # #         print(title,'::',isbn_series)
    # #         title_isbn_dict[title] = isbn_series.tolist()
    # #     return title_isbn_dict
    
    # # dict_unique_isbn = make_isbn_dict(final_df)
    # # with open('.\Single_ISBN.pickle', 'wb') as handle:
    # #     pickle.dump(dict_unique_isbn, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('.\Single_ISBN.pickle', 'rb') as handle:
    #     multiple_isbn_dict = pickle.load(handle)

    # # print(multiple_isbn_dict['Selected Poems'])
    # def add_unique_isbn_col(df):
    #     df['unique_isbn'] = df.apply(lambda row: multiple_isbn_dict[row.Book_Title][0] if row.Book_Title in multiple_isbn_dict.keys() else row.ISBN, axis=1)
    #     return df
    # final_df = add_unique_isbn_col(final_df)
    # # print(books_with_ratings.head())
    # # print(books_with_ratings[books_with_ratings.Book_Title=='Selected Poems'].head())

    # final_df.to_csv(r'.\\Project1\\Book-Crossing Dataset\\final_after.csv',index=False) #index=false去掉dataframe的第一欄
#----------------------------------------unique isbn opened if need--------------------------------------- 

#step7. Building matrix

# Create User-Item Sparse Matrix
# In a user-item sparse matrix, items’ values are present in the column, and users’ values are present in the rows. The rating of the user is present in the cell. 
# Such is a sparse matrix because there can be the possibility that the user cannot rate every movie items, and many items can be empty or zero.
    train_df = pd.read_csv(r'.\\Project1\\Book-Crossing Dataset\\final_after.csv') #直接read new csv file

    def set_pandas_display_options():
        display = pd.options.display
        display.max_columns = 15
        display.max_rows = 100
        display.max_colwidth = 199
        display.width = None
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# a. Cosine similarity
    #因為train_df太大to創建matrix，故需要刪減
    count1 = train_df['User-ID'].value_counts(ascending=False) #User-counts，User的評分分數
    matrix_df = train_df[train_df['User-ID'].isin(count1[count1>200].index)]
    count2 = train_df.groupby('unique_isbn')['Book_Rating'].count()#書的評分次數
    matrix_df = train_df[train_df['unique_isbn'].isin(count2[count2>100].index)]
    # count2 = train_df.groupby('User-ID').Book_Rating.sum() #User-rating-counts，User的評分次數
    # matrix_df = matrix_df[matrix_df['Book_Rating'].isin(count2[count2 > 20].index)]
    # matrix_df.to_csv(r'.\\Project1\\Book-Crossing Dataset\\matrix_df.csv',index=False)
    for isbn in matrix_df['unique_isbn']:
        if 'X' or 'x' in isbn: 
            isbn1 = re.sub('[A-Za-z]',"",isbn)
            matrix_df['unique_isbn'].replace(isbn,isbn1,inplace=True)
    # print(matrix_df['Country'])
    matrix_df['unique_isbn'] = matrix_df['unique_isbn'].astype(np.int64)
    matrix_df['Book_Rating'] = matrix_df['Book_Rating'].astype(np.int64)
    # 從這開始是input training data set:
    final_matrix = matrix_df.pivot_table(index='User-ID', columns='unique_isbn', values='Book_Rating')
    # print(final_matrix)
    # Pearson's 
    mean_ratings = final_matrix.mean(axis=0)
    final_matrix = final_matrix.sub(mean_ratings, axis=1)
    final_matrix.fillna(0,inplace=True)
    final_matrix = final_matrix.astype(np.int64)
    cosine_similarity_array = cosine_similarity(final_matrix) #這裡會用Consine
    set_pandas_display_options()
    # print(cosine_similarity_array)
    cosine_similarity_df = pd.DataFrame(cosine_similarity_array,index=final_matrix.index,columns=final_matrix.index)
    print(cosine_similarity_df) #稀鬆矩陣
    # 填入要預測的對象:
    predicted_user = 11676
    cosine_similarity_series = cosine_similarity_df.loc[predicted_user] 
    order = cosine_similarity_series.sort_values(ascending=False) #排出User-User之間的correlation 
    # print(order[(order>0.3)].head(15))
    print(order.head(15))
    # 推薦新書給預測的對象:
    Recommend_df = matrix_df.sort_values(by = ['Book_Rating'],ascending=False) #Rating先由大到小排序，才可推薦由高分到低分的新書
    Recommend_df.to_csv(r'.\\Project1\\Book-Crossing Dataset\\Recommend_df.csv',index=False)
    similar_user_list = []
    predicted_user_list = []
    for i in range(0,len(Recommend_df[Recommend_df['User-ID'] == predicted_user]['unique_isbn'])):
        predicted_user_list.append(Recommend_df[Recommend_df['User-ID'] == predicted_user]['unique_isbn'].values[i]) #預測對象已讀過的書
    
    for i in range(0,5):
        similar_user = order.iloc[1:6,].index[i] #int64index轉int要有.index[0]才可用，因int64index是為list
        similar_user_list.append(similar_user)

    def get_isbn_from_index(title, number, predicted_user_list):
        return Recommend_df[Recommend_df['unique_isbn'] == title]['User-ID'].values[0]
    def get_index_from_isbn(index, number, predicted_user_list):
        # return matrix_df[matrix_df['User-ID'] == index] #All
        re_books = Recommend_df[Recommend_df['User-ID'] == index]['unique_isbn'].values[number] #第[number]本出現的書
        if re_books in predicted_user_list:
            # print('Not match:',re_books)
            return 0
        return re_books
    for j in similar_user_list:
        print('Top similarity users:',j)
        for i in range(0,len(Recommend_df[Recommend_df['User-ID'] == j]['unique_isbn'])):
            if get_index_from_isbn(j, i, predicted_user_list) == 0:
                pass
            else:
                print('Recommend books ISBN:',get_index_from_isbn(j, i, predicted_user_list))


# --SVD--
    filter_df = matrix_df.drop(columns=['ISBN','Country','State','City','Book_Title','Book-Author','Year-Of-Publication','Publisher','Age'],axis=1)
    filter_df = filter_df.reindex(columns=['User-ID','unique_isbn','Book_Rating'])
    #使用surpise的train_test_split要reader
    reader = Reader(rating_scale=(1, 10)) #Rating range 1~10
    # Load the data into a 'Dataset' object directly from the pandas df.
    # Note: The fields must be in the order: user, item, rating
    data = Dataset.load_from_df(filter_df, reader)
    train_set, test_set = train_test_split(data, test_size=0.2)
    # 調RMSE方法1
    # model = SVD()
    # kf = KFold(n_splits=3)
    # for train_set, test_set in kf.split(data):
    #     model.fit(train_set)
    #     predictions = model.test(test_set)
    #     accuracy.rmse(predictions)

    # 調RMSE方法2
    with open("model.pkl", 'rb') as file:
        model = pickle.load(file)

    # param_grid = {'n_factors': [80, 100, 120], 'lr_all': [0.001, 0.005, 0.01], 'reg_all': [0.01, 0.02, 0.04]}
    # gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    # gs.fit(data)
    # # print(gs.best_score['rmse'])
    # # print(gs.best_params['rmse']) #Result: {'n_factors': 80, 'lr_all': 0.005, 'reg_all': 0.02}
    # model = SVD(n_factors=80, lr_all=0.005, reg_all=0.04)
    # model.fit(train_set) # re-fit on only the training data using the best hyperparameters
    # test_pred = model.test(test_set)
    # accuracy.rmse(test_pred, verbose=True)

    # with open('model.pkl', 'wb') as file:
    #     pickle.dump(model, file)
    

    # uid = 184532
    # iid = 671739433
    # pred1 = model.predict(uid, iid, verbose=True)

    # Start predicted 
    Ratings_testX = pd.read_csv(r'.\\Project1\\Book-Crossing Dataset\\Ratings_testX.csv')
    predicted_rating = []
    # for i,j in zip(Ratings_testX['User-ID'],Ratings_testX['ISBN']):
    for i,j in zip([188593,184532,198711],[553148001,671739433,394588320]):
        uid = i
        iid = j
        pred = model.predict(uid,iid)
        print(pred)
        predicted_rating.append(round(pred.est))
    # # Ratings_testX['Predicted Rating'] = predicted_rating
    # predict_csv = pd.DataFrame(predicted_rating, columns=None)
    # predict_csv.to_csv(r'.\\Project1\\Book-Crossing Dataset\\110209058.csv',index=False)

# --KNN--
    # model = KNNBasic()
    # model.fit(train_set)
    # uid = 188593
    # iid = 553148001
    # # 获取指定用户和电影的评级结果
    # pred2 = model.predict(uid, iid, r_ui=4, verbose=True)

# --Top recommendation books for each user--
    def get_top_n(predictions, n=10):
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]
            
        return top_n

    pred = model.test(test_set)
    top_n = get_top_n(pred)
    def get_reading_list(userid):
        reading_list = defaultdict(list)
        top_n = get_top_n(pred, n=10)
        for n in top_n[userid]:
            book, rating = n
            title = matrix_df.loc[matrix_df.unique_isbn==book].Book_Title.unique()[0]
            reading_list[title] = rating
        return reading_list
    # Just take a random look at user_id=60337
    example_reading_list = get_reading_list(85526)
    for book, rating in example_reading_list.items():
        # print(example_reading_list)
        print(f'{book}: {rating}')

# --End--






#-----------------------------------------------------------------------------------------------------------------------------------------------------------   
    # # KNeighborsRegressor
    # filter_df = matrix_df.drop(columns=['ISBN','Country','State','City','Book_Title','Book-Author','Year-Of-Publication','Publisher','Age'],axis=1)
    # filter_df_normalize=StandardScaler().fit_transform(filter_df)
    # print(filter_df)
    # filter_df_attributes =filter_df.drop("Book_Rating", axis =1)
    # filter_df_labels =filter_df['Book_Rating']
    # X_train, X_test, y_train, y_test = train_test_split(filter_df_attributes, filter_df_labels, test_size = 0.2)
    # k_fold = list(KFold(n_splits=5, shuffle=True).split(X_train, y_train))
    # print(k_fold)
    # def grid_search_best_model(model, params, k_fold, X_train, y_train):
    #     grid_search = GridSearchCV(model,params,cv=k_fold).fit(X_train,y_train)
    #     print("Best params", grid_search.best_params_)
    #     print("Best estimator", grid_search.best_estimator_)
    #     print("Best score:", grid_search.best_score_)
    
    #     return grid_search.best_estimator_
    # model_results = {}

    # def score_model(model,X_train, X_test, y_train, y_test,
    #             show_plot=True):   
    #     y_pred = model.predict(X_test)  
    #     print(f"Training score: {model.score(X_train,y_train)}")
    #     print(f"Test score: {r2_score(y_test, y_pred)}")
    #     print("MSE: ", mean_squared_error(y_test, y_pred))
        
    #     predictions_comparision = pd.DataFrame({'Actual': y_test.tolist(), 'Predicted': y_pred.tolist()}).sample(50)
    #     if show_plot == True:
    #         predictions_comparision.plot(kind="bar", figsize=(12,8),title="Actual vs predicted values")
    #     print(predictions_comparision.sample(50))    
        
        
    #     return {
    #         "training_score": model.score(X_train,y_train),
    #         "test_score_r2" : r2_score(y_test, y_pred),
    #         "test_score_mse" : mean_squared_error(y_test, y_pred)
    #     }

    # def compare_results():
    #     for key in model_results:
    #         print("Regression: ", key)
    #         print("Trainign score", model_results[key]["training_score"])
    #         print("R2 Test score ", model_results[key]["test_score_r2"])
    #         print("MSE Test score ", model_results[key]["test_score_mse"])
    #         print()

    # params={
    #     "n_neighbors": range(5, 30),
    #     "leaf_size":[20,30,50,70]
    # }

    # knn = grid_search_best_model(KNeighborsRegressor(), params, k_fold, X_train, y_train)
    # model_results["knn"] = score_model(knn, X_train, X_test, y_train, y_test)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
    # In KnearestNeighbor we are going to use the cosine similarity.
    # us_canada_user_rating = matrix_df[matrix_df['Country'].str.contains("usa|canada",na=False)] #Cannot mask with non-boolean array containing NA / NaN values, so need na=False
    # us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)
    # us_canada_user_rating = us_canada_user_rating.drop_duplicates(['User-ID', 'unique_isbn'])
    # us_canada_user_rating_pivot = us_canada_user_rating.pivot(index='User-ID', columns='unique_isbn', values = 'Book_Rating').fillna(0)
    # us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values) #把一個稀疏的np.array壓縮(csr=Row, csc=Columns)
    # model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    # model_knn.fit(us_canada_user_rating_matrix)
    # query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])
    # set_pandas_display_options()
    # # print(us_canada_user_rating_pivot)
    # distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)
    # for i in range(0, len(distances.flatten())): # Return a copy of the array collapsed into one dimension.(類似降維)
    #     if i == 0:
    #         print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
    #     else:
    #         print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
    # def getrecommend(user_name):
    #     distances,indices=model_knn.kneighbors(us_canada_user_rating_pivot.iloc[us_canada_user_rating_pivot.index==user_name, :].values.reshape(1,-1),n_neighbors=6)
    #     for i in range(0, len(distances.flatten())):
    #         if i == 0:
    #             print('Recommendations for {0}:\n'.format(user_name))
    #         else:
    #             print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
    # getrecommend(135265)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
    # # Setting global variables
    # global metric,k
    # k=10
    # metric='cosine'

    # #This function finds k similar users given the user_id and ratings matrix 
    # #These similarities are same as obtained via using pairwise_distances
    # def findksimilarusers(user_id, ratings, metric = metric, k=k):
    #     similarities=[]
    #     indices=[]
    #     model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
    #     model_knn.fit(ratings)
    #     loc = ratings.index.get_loc(user_id)
    #     distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = 6)
    #     similarities = distances.flatten() #1-distances.flatten()
                
    #     return similarities,indices

    # def predict_userbased(user_id, item_id, ratings, metric = metric, k=k):
    #     prediction=0
    #     user_loc = ratings.index.get_loc(user_id)
    #     item_loc = ratings.columns.get_loc(item_id)
    #     similarities, indices=findksimilarusers(user_id, ratings,metric, k) #similar users based on cosine similarity
    #     mean_rating = ratings.iloc[user_loc,:].mean() #to adjust for zero based indexing
    #     sum_wt = np.sum(similarities)-1
    #     product=1
    #     wtd_sum = 0 
        
    #     for i in range(0, len(indices.flatten())):
    #         if indices.flatten()[i] == user_loc:
    #             continue;
    #         else: 
    #             ratings_diff = ratings.iloc[indices.flatten()[i],item_loc]-np.mean(ratings.iloc[indices.flatten()[i],:])
    #             product = ratings_diff * (similarities[i])
    #             wtd_sum = wtd_sum + product
        
    #     #in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    #     #which are handled here as below
    #     prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    #     if prediction <= 0:
    #         prediction = 1   
    #     elif prediction >10:
    #         prediction = 10
        
    #     print(prediction)
    #     print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))

    #     return prediction
    # predict_userbased(11676,2258560,final_matrix);
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

def histogram(df):
    plt.figure(figsize=(8,8))
    hist = plt.hist(df, bins=30)#,orientation='horizontal')# = 倒反
    plt.ylabel('Count of items')
    plt.xlabel('Num of Rating')
    # plt.axis([0,12000,0,100000]) #設定xmin xmax ymix ymax
    plt.grid(True) #格線
    plt.show()

def countplot(df):
    sns.countplot(data=df, x=df.index, y=df.values)
    plt.show()

def heatplot(df):
    correlation= df.corr()
    column = df.columns
    plt.figure(figsize=(15,15))
    sns.heatmap(correlation,cbar=True,square=True,annot=True,fmt='.1f',annot_kws={'size': 15},
                xticklabels=column,yticklabels=column,alpha=0.7,cmap= 'coolwarm')
    plt.show()

def barplot(df,col,name):
    df_v=df[col].value_counts().head(25).reset_index()
    df_v.columns=[col,'count']
    plt.figure(figsize=(10,10))
    x = sns.barplot(x=col,y='count',data=df_v)
    # print(df_v[col].head(25).sort_values())
    x.set_xticklabels(labels=df_v[col].head(25))#,rotation = 90)
    plt.ylabel('count')
    plt.title(name,size=20)
    plt.show()

def barplot_partial(df):
    plt.figure(figsize=(10,10))
    # plt.plot(linewidth=2)
    plt.rc('font', size=8)
    sns.barplot(x=df.values,y=range(len(df.index)),data=df, orient='h')
    # print(df_v[col].head(25).sort_values())
    plt.yticks([9,8,7,6,5,4,3,2,1,0],df.index)#, rotation = 'vertical',)
    plt.ylabel('count')
    # plt.title(name,size=20)
    plt.show()

if __name__ == "__main__":
    main()