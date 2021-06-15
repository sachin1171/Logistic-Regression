#########################problem 1 ##########################
#load the data set
Affairs <- read.csv(file.choose())
install.packages('AER')
library('AER')
library(plyr)

affairs <- data("Affairs")
View(Affairs)

affairs1 <- Affairs
summary(affairs1)

table(affairs1$affairs)

affairs1$ynaffairs[affairs1$affairs > 0] <- 1
affairs1$ynaffairs[affairs1$affairs == 0] <- 0
affairs1$gender <- as.factor(revalue(Affairs$gender,c("male"=1, "female"=0)))
affairs1$children <- as.factor(revalue(Affairs$children,c("yes"=1, "no"=0)))

View(affairs1)

colnames(affairs1)

class(affairs1)

attach(affairs1)

# GLM function use sigmoid curve 
model <- glm(ynaffairs ~ factor(gender) + age+ yearsmarried+ factor(children) + religiousness+
               education+occupation+rating, data = affairs1,family = "binomial")

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Confusion matrix table 
prob <- predict(model,affairs1,type="response")
summary(model)

# We are going to use NULL and Residual Deviance to compare the between different models

# Confusion matrix and considering the threshold value as 0.5 
confusion<-table(prob>0.5,affairs1$ynaffairs)
confusion

# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
yes_no <- NULL

pred_values <- ifelse(prob>=0.5,1,0)
yes_no <- ifelse(prob>=0.5,"yes","no")

# Creating new column to store the above values
affairs1[,"prob"] <- prob
affairs1[,"pred_values"] <- pred_values
affairs1[,"yes_no"] <- yes_no

View(affairs1[,c(1,9:11)])

table(affairs1$ynaffairs,affairs1$pred_values)

library(ROCR)

rocrpred<-prediction(prob,affairs1$ynaffairs)
rocrperf<-performance(rocrpred,'tpr','fpr')

str(rocrperf)

plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))

rocr_cutoff <- data.frame(cut_off = rocrperf@alpha.values[[1]],fpr=rocrperf@x.values,tpr=rocrperf@y.values)
colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
View(rocr_cutoff)

library(dplyr)

rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off,6)
# Sorting data frame with respect to tpr in decreasing order 
rocr_cutoff <- arrange(rocr_cutoff,desc(TPR))

View(rocr_cutoff)
#################### problem 2 #######################
library(data.table)
av_ds<- read.csv(file.choose())
av_ds<- na.omit(av_ds)
View(av_ds)
attach(av_ds)
summary(av_ds)

colnames(av_ds)
plot(av_ds)
elec_res <- glm(Age ~ Area_Income+`Daily.Internet.Usage`+`Ad_Topic_Line`+City+Male+Country+Timestamp+`Clicked_on_Ad`, data = av_ds)
summary(elec_res)

library(MASS)
stepAIC(elec_res)

library(car)
vif(elec_res)

exp(coef(elec_res))


prob <- as.data.frame(predict(elec_res, type = c("response"), av_ds))
final <- cbind(av_ds,prob)
confusion <- table(prob>0.5, av_ds$Clicked_on_Ad)
table(prob>0.5)

confusion

Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy
####################problem 3 ##########################
library(data.table)
#load the data set
election_data <- read.csv(file.choose())
#omit the na values
election_data<- na.omit(election_data)
View(election_data)
attach(election_data)
summary(election_data)

colnames(election_data)
plot(election_data)
elec_res <- glm(Result ~ Year+`Amount.Spent`+`Popularity.Rank`, data = election_data)
summary(elec_res)

library(MASS)
stepAIC(elec_res)

library(car)
vif(elec_res)

exp(coef(elec_res))

#final model
prob <- as.data.frame(predict(elec_res, type = c("response"), election_data))
final <- cbind(election_data,prob)
confusion <- table(prob>0.5, election_data$Result)
table(prob>0.5)

confusion


Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy

sensitivity<-(1679/sum(1679+888)) # TPR = TP/TP+FN
sensitivity # 0.65

specificity<-(39034/(39034+3610)) # = TN/TN+FP
specificity # 0.915

precision<-(1679/(1679+3610))
precision # 0.317

a<-(sensitivity * precision)
b<-(sensitivity+precision)
F1<-(2*a/b)
F1 # 0.427

rocrpred<-prediction(prob,election_data$Result)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T)

#################################problem 4  ####################
library(readr)
library(ROCR)
bank<-read.csv(file.choose())
View(bank)
bank<-na.omit(bank)
View(bank)
summary(bank)
str(bank)
bank$day=as.factor(bank$day)
bank$campaign=as.factor(bank$campaign)
bank$pdays=as.factor(bank$pdays)
bank$previous=as.factor(bank$previous)
str(bank)
attach(bank)


fit1<-glm(y~ age+balance+housing+loan, family = "binomial", data = bank)
summary(fit1) # AIC= 31511

fit2<-glm(y~ age, family = "binomial", data = bank)
summary(fit2) # AIC= 32607

fit3<-glm(y~ loan+duration, data= bank, family = "binomial")
summary(fit3) # AIC= 27254

fit4<-glm(y~ age+balance+housing,family = "binomial", data = bank)
summary(fit4) # AIC= 31701

fit5<-glm(y~ age+housing+balance+duration+campaign, data= bank, family = "binomial")
summary(fit5) #AIC= 26001

fit6<-glm(y~ age+duration, data= bank, family = "binomial")
summary(fit6) #AIC= 27473


#USING fit5

exp(coef(fit5))
table(bank$y)
prob <- predict(fit5,type=c("response"),bank)
prob
confusion<-table(prob>0.5,bank$y)
probo <- prob>0.5
table(probo)
confusion

Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy # 0.90
Error <- 1-Accuracy
Error # 0.09

sensitivity<-(1679/sum(1679+888)) # TPR = TP/TP+FN
sensitivity # 0.65

specificity<-(39034/(39034+3610)) # = TN/TN+FP
specificity # 0.915

precision<-(1679/(1679+3610))
precision # 0.317

a<-(sensitivity * precision)
b<-(sensitivity+precision)
F1<-(2*a/b)
F1 # 0.427
