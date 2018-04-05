rm(list = ls()); 
#Loading the libraries
library(data.table);
library(xgboost);
library(Matrix);
library(caret)
library(ggplot2)
#Determining the target
target <- "readmitted";
#Reading the data
train <- read.csv("diabetic_data.csv", na.strings = "?");
#dropping the coloumns
train$encounter_id <- NULL; 
train$patient_nbr <- NULL;
train$medical_specialty = NULL
train$payer_code = NULL
train$weight = NULL
train$discharge_disposition_id = NULL
train$admission_source_id = NULL
#Data cleaning
#converting variables to factors
train$race <- as.numeric(as.factor(train$race));
# Providing distinct values 
train$age <- ifelse(train$age == "[0-10)",  0, train$age);
train$age <- ifelse(train$age == "[10-20)", 10, train$age);
train$age <- ifelse(train$age == "[20-30)", 20, train$age);
train$age <- ifelse(train$age == "[30-40)", 30, train$age);
train$age <- ifelse(train$age == "[40-50)", 40, train$age);
train$age <- ifelse(train$age == "[50-60)", 50, train$age);
train$age <- ifelse(train$age == "[60-70)", 60, train$age);
train$age <- ifelse(train$age == "[70-80)", 70, train$age);
train$age <- ifelse(train$age == "[80-90)", 80, train$age);
train$age <- ifelse(train$age == "[90-100)", 90, train$age);
train$age <- as.numeric(train$age);

train$gender <- as.numeric(as.factor(train$gender));
train$time_in_hospital <- as.numeric(train$time_in_hospital);
train$medical_specialty <- as.numeric(as.factor(train$medical_specialty));
train$num_lab_procedures <- as.numeric(train$num_lab_procedures);
train$num_procedures <- as.numeric(train$num_procedures);
train$num_medications <- as.numeric(train$num_medications);
train$number_outpatient <- as.numeric(train$number_outpatient);
train$number_emergency <- as.numeric(train$number_emergency);
train$number_inpatient <- as.numeric(train$number_inpatient);
train$diag_1 <- as.numeric(as.factor(train$diag_1));
train$diag_2 <- as.numeric(as.factor(train$diag_2));
train$diag_3 <- as.numeric(as.factor(train$diag_3));
train$number_diagnoses <- as.numeric(train$number_diagnoses);

train$max_glu_serum <- ifelse(train$max_glu_serum == "None",  0, train$max_glu_serum);
train$max_glu_serum <- ifelse(train$max_glu_serum == "Norm",  100, train$max_glu_serum);
train$max_glu_serum <- ifelse(train$max_glu_serum == ">200",  200, train$max_glu_serum);
train$max_glu_serum <- ifelse(train$max_glu_serum == ">300",  300, train$max_glu_serum);
train$max_glu_serum <- as.numeric(train$max_glu_serum);

train$A1Cresult <- ifelse(train$A1Cresult == "None",  0, train$A1Cresult);
train$A1Cresult <- ifelse(train$A1Cresult == "Norm",  5, train$A1Cresult);
train$A1Cresult <- ifelse(train$A1Cresult == ">7",    7, train$A1Cresult);
train$A1Cresult <- ifelse(train$A1Cresult == ">8",    8, train$A1Cresult);
train$A1Cresult <- as.numeric(train$A1Cresult);

columns <- c("metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
             "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
             "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
             "citoglipton", "insulin", "glyburide.metformin", "glipizide.metformin",
             "glimepiride.pioglitazone", "metformin.rosiglitazone", "metformin.pioglitazone");
for( c in columns ){
  train[[c]] <- ifelse(train[[c]] == "Up",     +10, train[[c]]);
  train[[c]] <- ifelse(train[[c]] == "Down",   -10, train[[c]]);
  train[[c]] <- ifelse(train[[c]] == "Steady", +0, train[[c]]);
  train[[c]] <- ifelse(train[[c]] == "No",     -20, train[[c]]);
  train[[c]] <- as.numeric(train[[c]]);
}
names(train)
train$change <- ifelse(train$change == "No", -1, train$change);
train$change <- ifelse(train$change == "Ch", +1, train$change);
train$change <- as.numeric(train$change);

train$diabetesMed <- ifelse(train$diabetesMed == "Yes", +1, train$diabetesMed);
train$diabetesMed <- ifelse(train$diabetesMed == "No",  -1, train$diabetesMed);
train$diabetesMed <- as.numeric(train$diabetesMed);
# 1 = [admitted withhin >30 or <30 days], 0 = not readamitted
train$readmitted <- ifelse(train$readmitted != "NO", 1, 0); # ">30", "<30", "NO"
train$readmitted <- as.numeric(train$readmitted);
train[] <- lapply(train, as.numeric);

# dropping Null values
library(tidyr)
train = train  %>%
  drop_na()
#splitting dataset
library(caTools)
set.seed(123)
split = sample.split(train$readmitted,SplitRatio = .8)
train_set = subset(train ,split == TRUE)
test_set = subset(train , split == FALSE)


train_new = train_set$readmitted
train_set$readmitted = NULL
test_new = test_set$readmitted
test_set$readmitted = NULL

#using XGBOOST PACKAGE TO PREDICT THE VALUES
dtrain <- xgb.DMatrix(as.matrix(train_set), label = train_new , missing = NA)
dtest <- xgb.DMatrix(as.matrix(test_set), label = test_new , missing = NA)
watchlist <- list(train = dtrain)

param <- list(
  objective           = "reg:logistic",
  booster             = "gbtree",
  eta                 = 0.02,
  max_depth           = 5,
  eval_metric         = "auc",
  min_child_weight    = 150,
  alpha               = 0.00,
  subsample           = 0.70,
  colsample_bytree    = 0.70,
  nrounds             = 20000,
  early_stepping_rounds = 100
)

set.seed(123)
clf <- xgb.cv(  params                = param,
                data                  = dtrain,
                nrounds               = 20000,
                verbose               = 1,
                watchlist             = watchlist,
                maximize              = TRUE,
                nfold                 = 5,
                nthread               = 4,
                print_every_n         = 50,
                stratified            = TRUE,
                early_stopping_rounds = 100
); 

 
print(clf)  
#PREDICTING THE  READMISSION RATE
model1 = xgb.train(data = dtrain,params= param ,nrounds = 2523)
preds=predict(model1,dtest)
pred_test=ifelse(preds > 0.5 ,1,0)
confusionMatrix(pred_test,test_new)
#VISUALIZATION OF IMPORTANT FACTORS
importance_matrix = xgb.importance(colnames(dtrain), model = model1)
xgb.plot.importance(importance_matrix)





