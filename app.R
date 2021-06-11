#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(naivebayes)
library(ggplot2)
library(dplyr)
library(psych)
library(rpart)
library(rpart.plot)
library(caret)
library(gbm)
library(class)
library(rattle)
library(e1071)

set.seed(123)
#Preloading the Data needed for the Outputs

#Data Input
rawData <- readRDS("data/rawData.rds")
rawDataWHO <- readRDS("data/rawDataWHO.rds")
clusteredWHO <- readRDS("data/clusteredWHO.rds")
klusteredWHO <- readRDS("data/klusteredWHO.rds")

#Transforming label into a factor on a 2nd dupe set
rawDataWF <- rawData
rawDataWF$label <- as.factor(rawDataWF$label)

#indexes for data splits
ind <- sample(2, nrow(rawData), replace = T, prob = c(0.75, 0.25)) 
ind2 <- sample(2, nrow(rawDataWHO), replace = T, prob = c(0.75, 0.25)) 
#Train/Test Data 
train1 <- rawData[ind ==1,]
trainw <- rawDataWHO[ind2 ==1,]
test <- rawData[ind ==2,]
testw <- rawDataWHO[ind2 ==2,]

#Test Data without label
test2 <- rawData[ind ==2, -1]

#Train/Test Data for factored label
trainf <- rawDataWF[ind ==1,]
testf <- rawDataWF[ind ==2,]

#Test Data without label
test2f <- rawDataWF[ind ==2, -1]

#Naive
naivemodel <- naive_bayes(trainf$label~., data = trainf)#, usekernel = T can perform better when numerical variables are not normally distributed
#Prediction 
p <- predict(naivemodel, test, type = "prob")
p1 <- predict(naivemodel, test[,-1])

#Pretty Confusion Matrix Function
draw_confusion_matrix <- function(cm) {
    
    layout(matrix(c(1,1,2)))
    par(mar=c(2,2,2,2))
    plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
    title('CONFUSION MATRIX', cex.main=2)
    
    # create the matrix 
    rect(150, 430, 240, 370, col='#3F97D0')
    text(195, 435, '0', cex=1.2)
    rect(250, 430, 340, 370, col='#551573')
    text(295, 435, '1', cex=1.2)
    text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
    text(245, 450, 'Actual', cex=1.3, font=2)
    rect(150, 305, 240, 365, col='#551573')
    rect(250, 305, 340, 365, col='#3F97D0')
    text(140, 400, '0', cex=1.2, srt=90)
    text(140, 335, '1', cex=1.2, srt=90)
    
    # add in the cm results 
    res <- as.numeric(cm$table)
    text(195, 400, res[1], cex=1.6, font=2, col='white')
    text(195, 335, res[2], cex=1.6, font=2, col='white')
    text(295, 400, res[3], cex=1.6, font=2, col='white')
    text(295, 335, res[4], cex=1.6, font=2, col='white')
    
    # add in the specifics 
    plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
    text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
    text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
    text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
    text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
    text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
    text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
    text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
    text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
    text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
    text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
    
    # add in the accuracy information 
    text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
    text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
    text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
    text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
} 
cmat <- confusionMatrix(p1, testf$label) #The matrix itself

#Did I even use this?
sampledData <- rawDataWF[sample(nrow(rawDataWF)),]
sampleTrain <- sampledData[ind == 1,]
sampleTest <- sampledData[ind == 2,]

#Random Forest and Simple Decision tree
trainctrl <- trainControl(method = "repeatedcv", number = 10, repeats =5)


rpart_tree <- train(label~.,
                    data = sampledData,
                    method = "rpart",
                    trControl = trainctrl)
rpart_tree2 <-train(tot_score~.,
                    data = rawDataWHO,
                    method = "rpart",
                    trControl = trainctrl)

rf_tree <- train(label~.,
                 data = rawDataWF,
                 method = "rf",
                 trControl = trainctrl)

rf_tree2 <- train(tot_score~.,
                  data = rawDataWHO,
                  method = "rf",
                  importance = TRUE,
                  trControl = trainctrl, )

resamps <- resamples (list (rpart_tree = rpart_tree, randomForest = rf_tree))
resamps2 <- resamples (list (rpart_tree = rpart_tree2, randomForest = rf_tree2)) 

V = caret::varImp(rf_tree2$finalModel)

#GBM
gbm_tree_WHO <- train(tot_score~., 
                      data = rawDataWHO,    #This is the WHO-5 one
                      method = "gbm", 
                      distribution = "gaussian", 
                      trControl = trainctrl, 
                      verbose = FALSE)

gbm_tree_auto <- gbm(formula = label ~ .,
                     distribution = "gaussian",
                     data = rawDataWF,
                     n.trees = 55,
                     interaction.depth = 3,
                     shrinkage = 0.05,
                     cv.folds = 5,
                     verbose = FALSE) 

predicted_values <- predict(gbm_tree_auto, test)
cmat5 = table(predicted_values, test[,1])
min_MSE <- which.min(gbm_tree_auto$cv.error)
knnPrediction <- knn(trainf[,2:3], testf[,2:3], trainf[,1], k = 7)

cmat3 <- confusionMatrix(knnPrediction, testf[,1])
scaledData <- scale(rawData)

#KMeans
dataWhoKlusters <- kmeans(scale(rawDataWHO[,2:6]), centers = 3, iter.max = 10, nstart = 150)
Kclusts = table(dataWhoKlusters$cluster, clusteredWHO$cluster)

#Hierarchical Clustering
rowsNo <- sample(nrow(rawDataWF))
ranData <- rawDataWF[rowsNo, 2:3 ]
ranDataClass <- rawDataWF[rowsNo, 1]
distance <- dist(rawDataWHO[,2:6], method = "euclidean") #counts euclidean distance between each variable and saves it here

fit<-hclust(distance, method = "complete")
Hgroups <- cutree(fit, 3)


# Define UI for application that draws a histogram
ui <- dashboardPage( skin = "red",
                     dashboardHeader(title = "Mental Health Illness Analysis"),
                     dashboardSidebar(
                         sidebarMenu(
                           h2("Data Sets"),
                             menuItem("DFSM Data", tabName = "dfsm", icon = icon("square")),
                             menuItem("WHO-5 Data", tabName = "who5", icon = icon("square")),
                           h2("Models"),
                             menuItem("GBM", tabName = "gbm", icon = icon("tree")),
                             menuItem("RF / SDT", tabName = "rf", icon = icon("tree")),
                             menuItem("Naive", tabName = "naive", icon = icon("flag")),
                             menuItem("KNN", tabName = "knn", icon = icon("flag")),
                             menuItem("Hierarchical", tabName = "hier", icon = icon("circle")),
                             menuItem("KMeans", tabName = "kmeans", icon = icon("circle")),
                           h2("Info"),
                             menuItem("About Me", tabName = "me", icon = icon("male"))
                         )
                     ),
                     dashboardBody(
                         tabItems(
                             tabItem("dfsm", fluidPage(
                               h1("Depression Data Gathered From Social Media"),
                               citFooter("Source: Rissola, Bahrainian and Crestani, 2020"),
                                dataTableOutput("dfsm_table"),
                                 shinydashboard::box(plotOutput("dfsm_plot"), width = 16),
                                 fluidRow(shinydashboard::box(selectInput("features", "Features:", c("polarity_score", "sadness_score", "wc" )), 
                                     width = 4, footer = "choose what feature to compare with label" ),
                                 shinydashboard::box(plotOutput("dfsmselect_plot"), width = 8) 
                            )))
                            ,
                             tabItem("who5", fluidPage(
                               h1("WHO-5 Synthetic Data Set"),
                               citFooter("Generated from following schema: https://www.mockaroo.com/cb872960"),
                               dataTableOutput("who5_table"),
                               shinydashboard::box(plotOutput("who5_plot"), width = 16, 
                               footer = "Pearson's Correlation Coefficient between all features, most important things to note are:
                                          a) The diagonal line that shows the correlation between q1 and q2, q2 and q3 etc,
                                          b) The correlations between q1 and the rest of the features,
                                          c) The correlations between tot_score and the rest of the features. 
                                          The answer generated to q1 is the most important answers that decides how the rest of the data will look like.")
                                     )
                             ),
                             tabItem("gbm", fluidPage(
                               shinydashboard::box(plotOutput("gbm_varImp_plot"), width = 16, 
                                                     footer = "This plot shows the importance of each feature. 
                                                     q1 = 57, q5 = 38, q2 = 27, q4 = 20, q3 = 17"),
                               shinydashboard::box(tableOutput("gbmWHO5"), width = 18)
                             )
                           ),
                             tabItem("rf", fluidPage(
                               h2("On DSFM Data Set"), fluidRow(
                                shinydashboard::box(title = "Simple Decision Tree Results", tableOutput("rpart_results"), footer = "Highest Accuracy is 0.81(0.806)"),
                                shinydashboard::box(title = "Random Forest Results", tableOutput("rf_results"), footer = "Highest Accuracy is 0.77(0.768)")),
                               h2("On WHO-5 Data Set"), fluidRow(
                                 shinydashboard::box(title = "Simple Decision Tree Results", tableOutput("rpart_results2"), footer = "Highest Rsquared is 0.88(0.883)"),
                                 shinydashboard::box(title = "Random Forest Results", tableOutput("rf_results2"), footer = "Highest Rsquared is 99 (0.987), the middle one")),
                               
                               
                               
                             )),
                             tabItem("naive", fluidPage(
                               shinydashboard::box(title = "Confusion Matrix with Accuracy and Kappa Results on DFSM Data", 
                                                   plotOutput("naiveMatrix"), width = 12),
                               shinydashboard::box(title = "Density Plot of Sadness_Score ", 
                                                   plotOutput("naiveDPlot"), 
                                                              footer = "Green line is for Label = 1,
                                                              Orange line is for Label = 2. This shows that the higher the sadness_score (the more intense the emotion is in the text), the higher the chance to be labelled 1", width = 12 )
                             )),
                             tabItem("knn", fluidPage(
                               shinydashboard::box(title = "K-Nearest Neighbour",
                                                   sliderInput("Knn", "K Number:", min = 1, max = 9, value = c(7)), 
                                                   footer = "Do avoid using even numbers." ),
                               shinydashboard::box(plotOutput("knnMatrix"), width = 12)
                             )),
                             tabItem("hier", fluidPage(
                               shinydashboard::box(title = "Hierarchical Clustering",
                                                   selectInput("method", "Agglomeration method used", c("single", "complete", "average", "mcquitty", "median", "centroid")),
                                                   ),
                               shinydashboard::box(sliderInput("cut", "Cut Tree into:", min = 1, max = 7, value = c(3))),
                               shinydashboard::box(plotOutput("hierDen"), width = 12),
                               shinydashboard::box(plotOutput("hierPlot"), width = 12)
                             )),
                             tabItem("kmeans",
                               shinydashboard::box(title = "K-Means", 
                                                   sliderInput("klusters", "Klusters", min = 1, max = 7, value = c(3)),
                                                   footer = "Choose amount of Clusters wanted"),
                               shinydashboard::box(plotOutput("kmeansPlot"), width = 12, heigh = 756 )
                             ),
                           tabItem("me",
                                   title= "About Me",
                                   shinydashboard::box(footer ="I'm a 21 years old Final Year Studen at Brunel University,
                                        and this is part of my Final Year Project, a demo of the algorithms
                                        used to analyse Mental Health data. It's quite the amateur job,
                                        but I've learned a lot from this experience. Thank you for checking this out.",
                                                       width = 12)
                                   )
                         )  
                     ))
 



# Define server logic required
server <- function(input, output) {
    
    
    
    rawData <- readRDS("data/rawData.rds")
    rawDataWHO <- readRDS("data/rawDataWHO.rds")
    clusteredWHO <- readRDS("data/clusteredWHO.rds")
    klusteredWHO <- readRDS("data/klusteredWHO.rds")
    
    #Outputs
    output$dfsm_plot <- renderPlot({
        pairs.panels(rawData)
        
    })
    output$dfsmselect_plot <- renderPlot({
        plot(rawData[[input$features]], rawData$label,
             xlab = "Label", ylab = "Features")
    })
    output$gbm_varImp_plot <- renderPlot(ggplot2::ggplot(V, aes(x=reorder(rownames(V),Overall), y=Overall)) +
                                           geom_point( color="blue", size=4, alpha=0.6)+
                                           geom_segment( aes(x=rownames(V), xend=rownames(V), y=0, yend=Overall), 
                                                         color='skyblue') +
                                           xlab('Variable')+
                                           ylab('Overall Importance')+
                                           theme_light() +
                                           coord_flip() )
    
    output$who5_plot <- renderPlot({pairs.panels(rawDataWHO)})
    output$dfsm_table <- renderDataTable(rawData)
    output$who5_table <- renderDataTable(rawDataWHO)
    output$gbmWHO5 <- renderTable({gbm_tree_WHO$results})
    output$rf_results <- renderTable({rf_tree$results})
    output$rpart_results <- renderTable ({rpart_tree$results})
    output$rf_results2 <- renderTable({rf_tree2$results})
    output$rpart_results2 <- renderTable ({rpart_tree2$results})
    
    output$naiveMatrix <- renderPlot(draw_confusion_matrix(cmat))
    output$naiveDPlot <- renderPlot(ggplot(rawDataWF, aes(x=sadness_score, color = label)) + 
                                      geom_density() )
    
    output$knnMatrix <- renderPlot(draw_confusion_matrix(confusionMatrix(knn(
        trainf[,2:3], testf[,2:3], trainf$label, k = input$Knn), testf$label)))
    
    output$kmeansPlot <- renderPlot(plot(rawDataWHO$tot_score, rawDataWHO$id, col = kmeans(scale(rawDataWHO[,2:6]), centers = input$klusters, iter.max = 10, nstart = 150)$cluster))
    
    fitReact <- reactive(hclust(distance, method = input$method))
    
    output$hierDen <- renderPlot(plot(fitReact()))
    HgroupsReact <- reactive(cutree(fitReact(), input$cut))
    output$hierPlot <- renderPlot(plot(rawDataWHO$tot_score, rawDataWHO$id, col = HgroupsReact()))
    
    
    
    
    }

# Run the application 
shinyApp(ui = ui, server = server)
