# Title: Edx Data Science: Capstone Project Movielens Total Code
# Weichen Zhou
# Date last edited: 01/07/2021
# Reference: Edx Data Science course notes 
  # and textbook "Introduction to Data Science Data Analysis and Prediction Algorithms with R" By Prof. Rafael A. Irizarry

# Overall description: this is the R script for the Project Movielens.
#                      The script will generate predicted movie rating and output the RMSE score. 



# Install necessary packages: 
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")


# Load packages
library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(ggplot2)
library(lubridate)


# Download Movielens 10m data
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Unzip the downloaded data and create the ratings data frame
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# Create the movie list
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# Convert the movie list to data frame and format the columns
# Note: I'm running R 4.0.3
movies <- as.data.frame(movies)  %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# Combine the rating data to the movie list
movielens <- left_join(ratings, movies, by = "movieId")

# Make sure the tables are joined correctly, had NA/s for title and genres when testing
head(movielens)


# Create a test set (edx) and validation set, require that the validation set contains 10% of data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


# Make sure the movies in the validation set are all included in the edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add the removed rows from the validation set back to the train set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove all unnecessary files
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Now we start exploring our training set edx and do some pre-processing
head(edx)

# Number of rows and columns in edx
nrow(edx)
ncol(edx)

# Number of movies in edx
N_movies <- edx %>% distinct(movieId) %>% summarise(n=n())
N_movies

# Number of users in edx
N_users <- edx %>% distinct(userId) %>% summarise(n=n())
N_users

# Observe that there's a year related to each movie in the title col, we can extract that
# Do the same thing for validation set.
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))


# Interpret the time stamp as the time the rating occurred, for both testing and validation set
edx <- edx %>% mutate(rating_year = year(as_datetime(timestamp)))
validation <-validation %>% mutate(rating_year = year(as_datetime(timestamp)))


# Fix incorrect data where rating happens before movie comes out
# Assume there's documentation error and the rating happened the same year as the movie came out.
error <- edx %>% filter(year > rating_year)
edx$rating_year <- ifelse(edx$rating_year<edx$year,edx$year,edx$rating_year)
validation$rating_year <- ifelse(validation$rating_year<validation$year,validation$year,validation$rating_year)

# Define a movie_age to be the number of years the movie has been out when it is rated
# Compute the movie age, time different between rating and production.
edx <- edx %>% mutate(movie_age = rating_year-year)
validation <- validation %>% mutate(movie_age = rating_year-year)


# Remove timestamp
edx <- edx %>% select(-timestamp)
validation <- validation %>% select(-timestamp)

# Starting here, the goal is to come up with a list of predictors for the model that predicts rating. 
#Possible predictors: Average rating for the movie, User effect,year the movie was produced, Movie age = rating_year-year, Genre

# Rating and movie age
# Number of rating and Age
edx %>% group_by(movie_age) %>% summarise(n=n()) %>%  ggplot(aes(movie_age,n)) + scale_y_log10() +geom_point() + ggtitle("Number of ratings by movie age")

# Average rating and age
age_avg <- edx %>% group_by(movie_age) %>% summarise(a_i = mean(rating))

# Plot for visualization:
age_avg %>% ggplot(aes(movie_age,a_i)) + geom_point() + ggtitle("Average rating by movie age")

# Note that very old movies don't have very low ratings, 
# this may be cause by the fact that very few people rate them.
age_avg %>% arrange(-movie_age)

# We use this piece of code to check the guess above
edx %>% filter(movie_age>80) %>% group_by(movie_age) %>% summarise(n=n())


# Rating and year(the year the movies were made)
# Number of ratings in each year
edx %>% group_by(year) %>%summarise(n=n()) %>%ggplot(aes(year,n)) + geom_point() +ggtitle("Number of rating by year")

# Average rating by year
year_avg <- edx %>% group_by(year) %>% summarise(r_i = mean(rating)) 
year_avg %>% ggplot(aes(year,r_i)) + geom_point() +ggtitle("Average rating by year")


# Rating and Genre
# Note: multiple genres for one movie 
# This piece of code here will take a couple of minutes to run.
# The run time is not crazily long (took around 7 min on my laptop) and it didn't crash anything.
edx_split <- edx %>% separate_rows(genres,sep="[\\|]")

# Number of rating for each genre
rating_num_genre <- edx_split %>% group_by(genres) %>% summarise(n=n()) %>% arrange(n)
rating_num_genre

# Average rating and Genres
edx_split %>% group_by(genres) %>% summarise(avg_rating_by_genre = mean(rating)) %>% ggplot(aes(reorder(genres,avg_rating_by_genre),avg_rating_by_genre)) + geom_bar(stat = "identity") +coord_flip()
# From plot, genres do not affect rating much, therefore will not consider it in model


# Gather year and age induced rating average info into table
edx <- edx %>% left_join(age_avg,by = "movie_age")  %>% left_join(year_avg,by="year") 

# Below we consider rating effect by movie average m_avg and rating by user u_avg
# Need Regularization here, as indicated in previous course material
# And we'll compute the RMSE
# Function for RMSE
RMSE <- function(rating_true, rating_pred){
  sqrt(mean((rating_true-rating_pred)^2))
}

# Consider a sequence of lambda's
lambdas <- seq(0,10,0.25)

# Model one: prediction is given by weighing in effect of individual movie, user effect, movie_age effect and year effect
# Prior to testing this model, note that the following code shows
# that the year and movie_age are heavily negatively correlated,
edx %>% summarise(cor = cor(year,movie_age))

# This indicate that including both as predictors may not be a good idea.
# But we'll compute the RMSE to see
# Compute RMSE with different lambdas and store RMSE results
rmses <-sapply(lambdas,function(lambda){
  # Average mu of the given rating observations in the test set. 
  mu <- mean(edx$rating)
  # Rating and effect by movie average, with regularization
  movie_avg<- edx %>% group_by(movieId) %>%
    summarise(m_avg = sum(rating-mu)/(n()+lambda)) 
  # Average rating by user, with regularization
  user_avg<- edx %>% left_join(movie_avg,by="movieId") %>%
    group_by(userId) %>%
    summarise(u_avg = sum(rating-(mu+m_avg+a_i+r_i))/(n()+lambda))
  pred_data <- validation %>% left_join(movie_avg,by="movieId") %>%
    left_join(user_avg,by = "userId") %>%
    left_join(year_avg,by = "year") %>%
    left_join(age_avg,by="movie_age") 
  # Prediction
  rating_pred_data <- pred_data %>% mutate(pred = mu+m_avg+u_avg + a_i+r_i) 
  rating_pred <- rating_pred_data$pred
  # Compute the RMSE using function defined above
  RMSE(validation$rating,rating_pred)
})

# plot the lambda values and rmses
qplot(lambdas,rmses)

# find the optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

rmse <- min(rmses)
rmse
#min rmse is 0.93, not so good

# Model 2: since year and movie_age are heavily correlated, we try by removing one
# This model counts in movie effect, user effect and movie_age effect
# same lambda sequence, just re-initializing
lambdas <- seq(0,10,0.25)

# Compute RMSE with different lambdas and store RMSE results
rmses <-sapply(lambdas,function(lambda){
  # Average mu of the given rating observations in the test set. 
  mu <- mean(edx$rating)
  # Rating and effect by movie average, with regularization
  movie_avg<- edx %>% group_by(movieId) %>%
    summarise(m_avg = sum(rating-mu)/(n()+lambda)) 
  # Average rating by user, with regularization
  user_avg<- edx %>% left_join(movie_avg,by="movieId") %>%
    group_by(userId) %>%
    summarise(u_avg = sum(rating-(mu+m_avg+a_i))/(n()+lambda))
  pred_data <- validation %>% left_join(movie_avg,by="movieId") %>%
    left_join(user_avg,by = "userId") %>%
    left_join(year_avg,by = "year") %>%
    left_join(age_avg,by="movie_age") 
  # Prediction
  rating_pred_data <- pred_data %>% mutate(pred = mu+m_avg+u_avg + a_i) 
  rating_pred <- rating_pred_data$pred
  # Compute the RMSE using function defined above
  RMSE(validation$rating,rating_pred)
})

# plot the lambda values and rmses
qplot(lambdas,rmses)

# find the optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

rmse <- min(rmses)
rmse
# We see the minimum rmse here is 0.88, better than previous, but still not good enough

# Recall this plot:
age_avg %>% ggplot(aes(movie_age,a_i)) + geom_point() + ggtitle("Average rating by movie age")
# The average rating doesn't seem to affect too much by age, 
# the range where most points are is about 0.4
# Try with model removing movie age: 
# Model three: only counting in movie effect and user effect with regularization

# Same lambda sequence, just re-initializing
lambdas <- seq(0,10,0.25)

# Compute RMSE with different lambdas and store RMSE results
rmses <-sapply(lambdas,function(lambda){
  # Average mu of the given rating observations in the test set. 
  mu <- mean(edx$rating)
  # Rating and effect by movie average, with regularization
  movie_avg<- edx %>% group_by(movieId) %>%
    summarise(m_avg = sum(rating-mu)/(n()+lambda)) 
  # Average rating by user, with regularization
  user_avg<- edx %>% left_join(movie_avg,by="movieId") %>%
    group_by(userId) %>%
    summarise(u_avg = sum(rating-(mu+m_avg))/(n()+lambda))
  # Join the predictors m_avg and u_avg to the validation set                           
  pred_data <- validation %>% left_join(movie_avg,by="movieId") %>%
    left_join(user_avg,by = "userId") 
  # Prediction
  rating_pred_data <- pred_data %>% mutate(pred = mu+m_avg+u_avg) 
  rating_pred <- rating_pred_data$pred
  # Compute the RMSE using function defined above
  RMSE(validation$rating,rating_pred)
})

# plot the lambda values and rmses
qplot(lambdas,rmses)

# find the optimal lambda
lambda <- lambdas[which.min(rmses)]
lambda

rmse <- min(rmses)
rmse
# Here, the minimum rmse is 0.864817.
# Happy about this result. 
# Use model 3 and lambda = 5.25. 







