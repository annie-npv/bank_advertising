library(tidyverse)

# Set directory
setwd("/Users/annienguyen/STA467/Final project")

# Read the bank data file
bank_data <- read.csv("bank-additional-full.csv",header = TRUE, sep = ';')

# Change the columns name 
colnames(bank_data)[colnames(bank_data) == "loan"] <- "personal"
colnames(bank_data)[colnames(bank_data) == "y"] <- "deposit"

# Data dimensions and structure
str(bank_data)
dim(bank_data)

# Check for missing values in the entire data frame
any(is.na(bank_data))

# EDA

head(bank_data)
# Extract numeric variables from the bank_data dataset
numeric_data <- bank_data[, sapply(bank_data, is.numeric)]

# Calculate the correlation matrix
bank_corr <- cor(numeric_data)

# Load required libraries
library(reshape2)

# Convert the correlation matrix to a data frame
cor_df <- as.data.frame(bank_corr)
cor_df <- cbind(rownames(cor_df), cor_df)
colnames(cor_df)[1] <- "Variable1"
cor_df <- melt(cor_df, id.vars = "Variable1")

# Plot the correlation matrix using ggplot
ggplot(data = cor_df, aes(x = Variable1, y = variable, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "yellow", mid = "blue", high = "red", midpoint = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))


# Based on the correlation plot:
### nr.employed and euribor3m are highly positive correlated.
### nr.employed and emp.var.rate are highly positive correlated.
### euribor3m and emp.var.rate are highly positive correlated.


pdf("barplots.pdf", width = 12, height = 12)

# Create bar plots for each column
par(mfrow = c(5, 2))  # 5 rows, 2 columns
for (col in colnames(bank_data)) {
  if (is.numeric(bank_data[[col]])) {
    # For numeric columns, create a histogram
    hist(bank_data[[col]], main = col, xlab = col, col = "#00A36C")
  } 
}
dev.off()


# Plot the target variable y
library(gridExtra)

## Create a new plot for frequency of deposit
plot1 <- ggplot(bank_data, aes(x = deposit, fill = deposit)) +
  geom_bar() +
  labs(title = "Frequency of each Term Deposit Status") +
  theme_minimal()+
  theme(legend.position = "none")


# Create a new plot for percentage of deposit using a pie chart
plot2 <- ggplot(bank_data, aes(x = "", fill = deposit)) +
  geom_bar(width = 1) +
  coord_polar("y", start = 0) +
  labs(title = "Percentage of each Term Deposit Status") +
  theme_minimal()+
  theme(legend.position = "none")


# Print the plots side by side
grid.arrange(plot1, plot2, ncol = 2)


# Data preprocessing
# Month and day of week is not very important, so we drop those
bank_data <- bank_data[, !(names(bank_data) %in% c("month", "day_of_week"))]

# Apply preprocessing to the data
# Convert categorical variables to factors
bank_data$job <- factor(bank_data$job)
bank_data$marital <- factor(bank_data$marital, levels = c("divorced", "married", "single", "unknown"))
bank_data$default <- factor(bank_data$default, levels = c("no", "yes", "unknown"))
bank_data$housing <- factor(bank_data$housing, levels = c("no", "yes", "unknown"))
bank_data$personal <- factor(bank_data$personal, levels = c("no", "yes", "unknown"))
bank_data$contact <- factor(bank_data$contact, levels = c("cellular", "telephone"))
bank_data$poutcome <- factor(bank_data$poutcome, levels = c("failure", "nonexistent", "success"))

# Convert ordinal variable 'education' to ordered factor with custom levels
bank_data$education <- ordered(
  bank_data$education,
  levels = c("illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course", "university.degree", "unknown")
)

# Prepare the target variable
bank_data$deposit <- ifelse(bank_data$deposit == "yes", 1, 0)



# Machine Learning
set.seed(05102023)
data <- data %>% sample_frac(0.5)
trainIndex <- sample(nrow(data), 0.8*nrow(data))
train <- data[trainIndex, ]
test <- data[-trainIndex, ]
