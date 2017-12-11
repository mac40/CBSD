library(caret)
setwd("C:/Users/PC/Dropbox/data science/Cognitive behavior and social data/prova")
heart_df <- read.csv("heart_tidy.csv", sep = ',', header = FALSE)
str(heart_df)
head(heart_df)
set.seed(3033)
intrain <- createDataPartition(y = heart_df$V14, p= 0.7, list = FALSE)
training <- heart_df[intrain,]
testing <- heart_df[-intrain,]
dim(training)
dim(testing)
anyNA(heart_df)
summary(heart_df)
training[["V14"]] = factor(training[["V14"]])
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3233)

svm_Linear <- train(V14 ~., data = training, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
test_pred <- predict(svm_Linear, newdata = testing)
confusionMatrix(test_pred, testing$V14)
grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
set.seed(3233)
svm_Linear_Grid <- train(V14 ~., data = training, method = "svmLinear",
                         trControl=trctrl,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
svm_Linear_Grid
plot(svm_Linear_Grid)
test_pred_grid <- predict(svm_Linear_Grid, newdata = testing)
confusionMatrix(test_pred_grid, testing$V14 )
library(ROCR)
pred_1 = prediction(as.numeric(test_pred),testing$V14)
risultato_1 <- performance( pred_1 , "sens", "spec")
plot(risultato_1 , colorize=TRUE , main = "V14")
auc_1 = risultato_1@y.values[[1]][2]
auc_1

pred_2 = prediction(as.numeric(test_pred_grid),testing$V14)
risultato_2 <- performance( pred_2 , "sens", "spec")
plot(risultato_2 , colorize=TRUE , main = "V14")
auc_2 = risultato_2@y.values[[1]][2]
auc_2

conf = confusionMatrix(test_pred, testing$V14)$table
write.table(conf,"prova.csv")
