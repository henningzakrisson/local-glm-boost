# Step 0: Install necessary libraries
if(!require(data.table)) {
  install.packages("data.table", dependencies=TRUE)
  library(data.table)
}

# Step 1: Load necessary libraries
library(data.table)

# Step 2: Download the file from the URL
url <- "https://people.math.ethz.ch/~wueth/Lecture/freMTPL2freq.rda"
destfile <- "freMTPL2freq.rda"
download.file(url, destfile)

# Step 3: Load the data into R
load(destfile)
dat <- freMTPL2freq

# Step 4: Set RNG version and seed, then split the data into two datasets
RNGversion("3.5.0")
set.seed(500)
ll <- sample(c(1:nrow(dat)), round(0.9*nrow(dat)), replace = FALSE)
learn <- dat[ll,]
test <- dat[-ll,]

# Step 5: Save the datasets as CSV files
if(!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
write.csv(learn, file.path(output_dir, "train_data.csv"))
write.csv(test, file.path(output_dir, "test_data.csv"))

# Step 6: Delete the .rda file
file.remove(destfile)
