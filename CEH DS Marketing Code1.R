# Download data from kaggle: https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python/downloads/customer-segmentation-tutorial-in-python.zip/1
# If you don't have any of the packages used in this script type install.packages("name of package") in the console (e.g. install.packages("arules"))
# Setup ----
library(arules);library(arulesViz);library(dplyr);library(ggplot2);library(purrr)
data1 <- read.csv("~/Dropbox/Mall_Customers.csv")
names(data1)[[4]] <- "Annual_Income"
names(data1)[[5]] <- "Spending_Score"

# Check data for skewness in variables of interest ----
par(mfrow=c(2,2))
hist(as.numeric(data1$Gender))
hist(data1$Age)
hist(data1$Annual_Income)
hist(data1$Spending_Score)
dev.off()
data <- discretizeDF(data1)
as.tbl(data)

# Build transactions and association rules ----
trans <- as(data[,-1],"transactions")
summary(trans)
inspect(trans[1:5])
rules <- apriori(trans,parameter = list(supp = 0.02, conf = 0.1, target = "rules", minlen=4))
summary(rules)
inspect(rules[1:5])
inspect(sort(rules,by="lift")[1:10])
plot(rules)

## Change axes if you want:
# plot(rules, measure=c("support","lift"), shading="confidence");

atable <- as(rules,"data.frame")
write.csv(atable,"~/Dropbox/Mallcustomer_rules_byLift.csv") # write to disk

hist(atable$lift) # check for skewness
rhspin <- "Age"
rules.sub <- subset(rules, subset = rhs %pin% paste0(rhspin,"=") & lift>1)
plot(rules.sub)
inspect(rules.sub, by="support")

# remove redundant rules
# subsetRules <- which(colSums(is.subset(rules, rules)) > 1) # get subset rules in vector
# length(subsetRules)
# rules2 <- rules[-subsetRules] # remove subset rules.

# find what factors influenced rhs
rules.F <- apriori (data=trans, parameter=list (supp=0.001,conf = 0.08), appearance = list (default="lhs",rhs="Gender=Female"), control = list (verbose=F)) # get rules that lead to gender F
rules_conf <- sort (rules.F, by="confidence", decreasing=TRUE) # 'high-confidence' rules.
inspect(head(rules_conf))

rules.income <- apriori (data=trans, parameter=list (supp=0.001,conf = 0.15,minlen=2), appearance = list(default="rhs",lhs="Annual_Income=[72,137]"), control = list (verbose=F)) # those who bought 'milk' also bought..
rules_conf2 <- sort (rules.income, by="confidence", decreasing=TRUE) # 'high-confidence' rules.
inspect(head(rules_conf2))

# Scatterplot - 4 dimensions ----
ggplot(data1, aes(Age, Annual_Income, size=Spending_Score, color=Gender)) + geom_point() + theme_minimal()

# Cluster ----
library(cluster)
data1$Gender<-as.numeric(data1$Gender)
fit <- kmeans(data1,5)
clusplot(data1, fit$cluster,line=0, color=T, labels=T)
pca.data <- princomp(data1)
plot(pca.data)
data1$cluster <- as.factor(fit$cluster)
centers=as.data.frame(fit$centers)

# Plot clusters
g = ggplot(data1, aes(Age, Annual_Income, size=Spending_Score, shape=cluster,color=cluster)) + geom_point() + theme_minimal()

# Add centroids
g+geom_point(data=centers,aes(x = Age,y = Annual_Income),fill="black",size=5, inherit.aes = F)+
  geom_text(data = centers, aes(x = Age,y = Annual_Income-5, label=paste0("Cluster ",1:5," Centroid")), size=3, inherit.aes = F)

# Centroids only
ggplot(centers, aes(x = Age,y = Annual_Income))+geom_point()

# Classification ----

library(caret)
set.seed(123)
samp=sample(nrow(data1), .5*nrow(data1)) #idx of training set
knn_model <- knn3Train(train = data1[samp,-c(1,6)],test = data1[-samp,-c(1,6)],k = 5, cl = data1$cluster[-samp], prob = T)
confusionMatrix(data = data1$cluster[-samp], as.factor(as.numeric(knn_model)))
