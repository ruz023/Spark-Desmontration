
# Load SparkR library
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

# Start Spark session
sparkR.session(master="local[*]", sparkConfig=list(spark.driver.memory="2g"))

# Read data into a Spark dataframe
# sdf <- read.df("./winequality-white.csv", "csv", header="true", inferSchema="true") 
sdf <- read.df("/srv/jupyterhub/read-only/data/winequality-white.csv", "csv", header="true", inferSchema="true") 

# Cache dataframe
cache(sdf)

# Examine schema
schema(sdf)

# Split into train & test sets
seed <- 12345
train_df <- sample(sdf, withReplacement=FALSE, fraction=0.7, seed=seed)
test_df <- except (sdf, train_df)
dim(train_df)
dim(test_df)

# Train RF model
model <- spark.randomForest(train_df, taste ~ ., type="classification", numTrees=30, seed=seed)
head(summary(model))

# Predict on test data
predictions <- predict(model, test_df)
prediction_df <- collect(select(predictions, "id", "prediction"))

# Evaluate
library(dplyr)

actual_vs_predicted <- dplyr::inner_join(as.data.frame(sdf), prediction_df, "id") %>%
    dplyr::select (id, actual=taste, predicted=prediction)
mean(actual_vs_predicted$actual == actual_vs_predicted$predicted)
table(actual_vs_predicted$actual, actual_vs_predicted$predicted)

# Examine first few rows of actual vs predicted wine quality values
head(actual_vs_predicted)

# Save model (NOTE:  path must not exist)
write.ml(model, "wine-RF-model")
