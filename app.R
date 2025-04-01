if(!require("shiny"))
  install.packages("shiny")
if(!require("shinydashboard"))
  install.packages("shinydashboard")
if(!require("ggplot2"))
  install.packages("ggplot2")
if(!require("plotly"))
  install.packages("plotly")
if(!require("class"))
  install.packages("class")
if(!require("gmodels"))
  install.packages("gmodels")
if(!require("caret"))
  install.packages("caret")
if(!require("rpart"))
  install.packages("rpart")
if(!require("rpart.plot"))
  install.packages("rpart.plot")
if(!require("randomForest"))
  install.packages("randomForest")


# Load Libraries
library(shiny)
library(shinydashboard)
library(ggplot2)
library(class)  # For KNN
library(gmodels)  # For CrossTable
library(caret)  # For confusionMatrix
library(rpart)  # For Decision Tree
library(rpart.plot)  # For Decision Tree Plot
library(randomForest)  # For Random Forest

# UI Section
ui <- dashboardPage(
  dashboardHeader(title = "Mushroom Analysis"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("KNN Results", tabName = "knn", icon = icon("project-diagram")),
      menuItem("Decision Tree", tabName = "decision_tree", icon = icon("tree")),
      menuItem("K-Means Clustering", tabName = "kmeans", icon = icon("chart-line")),
      menuItem("Random Forest Model", tabName = "random_forest_model", icon = icon("random")),
      menuItem("Model Comparison", tabName = "model_comparison", icon = icon("balance-scale"))
    )
  ),
  dashboardBody(
    tabItems(
      # Tab for KNN
      tabItem(tabName = "knn",
              fluidRow(
                box(title = "KNN Results Table", width = 6, tableOutput("knn_table")),
                box(title = "Accuracy Metrics", width = 6, verbatimTextOutput("knn_accuracy"))
              ),
              fluidRow(
                box(title = "Confusion Matrix Heatmap", width = 12, plotOutput("knn_confusion_plot"))
              ),
              fluidRow(
                box(title = "Classification Bar Chart", width = 12, plotOutput("knn_bar_chart"))
              )
      ),
      # Tab for Decision Tree
      tabItem(tabName = "decision_tree",
              fluidRow(
                box(title = "Decision Tree Plot", width = 12, plotOutput("decision_tree_plot")),
                box(title = "Confusion Matrix", width = 12, tableOutput("decision_tree_confusion"))
              ),
              fluidRow(
                box(title = "Decision Tree Accuracy", width = 12, verbatimTextOutput("decision_tree_accuracy"))
              )
      ),
      # Tab for K-Means Clustering
      tabItem(tabName = "kmeans",
              fluidRow(
                box(title = "Cluster Dendrogram", width = 12, plotOutput("dendrogram_plot"))
              ),
              fluidRow(
                box(title = "Cluster Size Bar Chart", width = 12, plotOutput("bar_chart"))
              )
      ),
      # Tab for Random Forest Model
      tabItem(tabName = "random_forest_model",
              fluidRow(
                box(title = "Random Forest Confusion Matrix", width = 12, tableOutput("random_forest_confusion"))
              ),
              fluidRow(
                box(title = "Random Forest Accuracy", width = 12, verbatimTextOutput("random_forest_accuracy"))
              ),
              fluidRow(
                box(title = "Random Forest Error Plot", width = 12, plotOutput("random_forest_error_plot"))
              ),
              fluidRow(
                box(title = "Variable Importance Plot", width = 12, plotOutput("random_forest_importance_plot"))
              )
      ),
      # Tab for Model Comparison
      tabItem(tabName = "model_comparison",
              fluidRow(
                box(title = "Model Accuracy Comparison", width = 12, plotOutput("model_comparison_plot"))
              )
      )
    )
  )
)

# Server Section
server <- function(input, output) {
  
  # Load dataset for KNN
  mushrooms_format_knn <- read.csv('mushrooms_format_values.csv')
  mushrooms_format_knn_remove <- mushrooms_format_knn[-1]
  n_rows <- nrow(mushrooms_format_knn)
  total_70_percent <- floor(0.7 * n_rows)
  mushrooms_knn_train <- mushrooms_format_knn_remove[1:total_70_percent, ]
  mushrooms_knn_train_labels <- mushrooms_format_knn[1:total_70_percent, 1]
  values_70 <- total_70_percent + 1
  mushrooms_knn_test <- mushrooms_format_knn_remove[values_70:n_rows, ]
  mushrooms_knn_test_labels <- mushrooms_format_knn[values_70:n_rows, 1]
  
  # Perform KNN
  mushrooms_format_knn_pre <- knn(
    train = mushrooms_knn_train,
    test = mushrooms_knn_test,
    cl = mushrooms_knn_train_labels,
    k = 21
  )
  
  # Confusion Matrix for KNN
  aa <- table(mushrooms_knn_test_labels, mushrooms_format_knn_pre)
  confusion_matrix_result <- confusionMatrix(aa)
  
  # KNN Outputs
  output$knn_table <- renderTable({
    as.data.frame.matrix(aa)
  })
  
  output$knn_accuracy <- renderPrint({
    paste(
      "Accuracy: ", round(confusion_matrix_result$overall["Accuracy"], 4)
    )
  })
  
  # KNN Heatmap for Confusion Matrix
  output$knn_confusion_plot <- renderPlot({
    heatmap_data <- as.data.frame(as.table(aa))
    colnames(heatmap_data) <- c("Actual", "Predicted", "Frequency")
    
    ggplot(heatmap_data, aes(x = Predicted, y = Actual, fill = Frequency)) +
      geom_tile(color = "white") +
      scale_fill_gradient(low = "blue", high = "red") +
      labs(
        title = "Confusion Matrix Heatmap",
        x = "Predicted Labels",
        y = "Actual Labels"
      ) +
      theme_minimal()
  })
  
  # Classification Bar Chart for KNN
  output$knn_bar_chart <- renderPlot({
    correct <- sum(diag(aa))
    incorrect <- sum(aa) - correct
    
    bar_data <- data.frame(
      Classification = c("Correct", "Incorrect"),
      Count = c(correct, incorrect)
    )
    
    # aes -  It is used to define how data variables are mapped to visual properties of a plot, such as:
    
    # X-axis (x) and Y-axis (y)
    # Color (color)
    # Size (size)
    # Shape (shape)
    # Fill (fill)
    
    
    # alpha - Plot Transparency
    ggplot(bar_data, aes(x = Classification, y = Count, fill = Classification)) +
      geom_bar(stat = "identity", alpha = 0.8) + 
      scale_fill_manual(values = c("green", "red")) +
      labs(
        title = "Classification Results",
        x = "Classification Type",
        y = "Count",
        fill = "Type"
      ) +
      theme_minimal()
  })
  
  # Decision Tree Logic
  mushroom_format_decision <- read.csv('mushrooms_format_values.csv')
  mushroom_format_decision_train <- mushroom_format_decision[1:total_70_percent, ]
  values <- total_70_percent + 1
  mushroom_format_decision_test <- mushroom_format_decision[values:n_rows, ]
  
  target <- class ~ cap.shape + cap.surface + cap.color + bruises + odor + gill.attachment + gill.spacing + gill.size + gill.color + stalk.shape + stalk.root + stalk.surface.above.ring + stalk.surface.below.ring + stalk.color.above.ring + stalk.color.below.ring + veil.type + veil.color + ring.number + ring.type + spore.print.color + population + habitat
  
  decision_tree <- rpart(target, data = mushroom_format_decision_train, method = "class")
  decision_predictions <- predict(decision_tree, mushroom_format_decision_test, type = "class")
  confusion_matrix <- table(mushroom_format_decision_test$class, decision_predictions)
  
  accuracy_decision_tree <- sum(decision_predictions == mushroom_format_decision_test$class) / length(mushroom_format_decision_test$class)
  
  # Decision Tree Outputs
  output$decision_tree_plot <- renderPlot({
    rpart.plot(decision_tree)
  })
  
  output$decision_tree_confusion <- renderTable({
    as.data.frame.matrix(confusion_matrix)
  })
  
  output$decision_tree_accuracy <- renderPrint({
    paste("Accuracy: ", round(accuracy_decision_tree, 4))
  })
  
  # K-Means Clustering Logic
 mushrooms_format_k_means_clustering <- read.csv('mushrooms_format_values.csv')
  mushrooms_format_k_means_clustering <- mushrooms_format_k_means_clustering[-1]
  
  d <- dist(mushrooms_format_k_means_clustering, method = "euclidean")
  hfit <- hclust(d)
  grps <- cutree(hfit, k = 2)
  
  # K-Means Outputs
  output$dendrogram_plot <- renderPlot({
    plot(hfit)
    rect.hclust(hfit, k = 2, border = "red")
  })
  
  output$bar_chart <- renderPlot({
    cluster_sizes <- table(grps)
    bar_data <- data.frame(Cluster = names(cluster_sizes), Size = as.vector(cluster_sizes))
    
    ggplot(bar_data, aes(x = Cluster, y = Size, fill = Cluster)) +
      geom_bar(stat = "identity", alpha = 0.8) +
      labs(title = "Cluster Size Distribution", x = "Cluster", y = "Size") +
      theme_minimal()
  })
  
  # Random Forest Logic
 mushrooms_format_random <- read.csv('mushrooms_format_values.csv')
  mushrooms_format_random[] <- lapply(mushrooms_format_random, factor)
  
  mushrooms_format_random_train <- mushrooms_format_random[1:total_70_percent, ]
  mushrooms_format_random_test <- mushrooms_format_random[values_70:n_rows, ]
  
  rf_model <- randomForest(class ~ ., data = mushrooms_format_random_train, importance = TRUE)
  rf_prediction <- predict(rf_model, newdata = mushrooms_format_random_test)
  rf_confusion <- confusionMatrix(rf_prediction, mushrooms_format_random_test$class)
  
  # Random Forest Outputs
  output$random_forest_confusion <- renderTable({
    rf_confusion$table
  })
  
  output$random_forest_accuracy <- renderPrint({
    paste("Accuracy: ", round(rf_confusion$overall["Accuracy"], 4))
  })
  
  output$random_forest_error_plot <- renderPlot({
    plot(rf_model)
  })
  
  output$random_forest_importance_plot <- renderPlot({
    varImpPlot(rf_model)
  })
  
  # Model Comparison Outputs
  output$model_comparison_plot <- renderPlot({
    model_data <- data.frame(
      Model = c("KNN", "Decision Tree", "Random Forest"),
      Accuracy = c(
        round(confusion_matrix_result$overall["Accuracy"], 4),
        round(accuracy_decision_tree, 4),
        round(rf_confusion$overall["Accuracy"], 4)
      )
    )
    
    ggplot(model_data, aes(x = Model, y = Accuracy, fill = Model)) +
      geom_bar(stat = "identity", alpha = 0.8) +
      labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
      theme_minimal()
  })
}

# Run the App
shinyApp(ui, server)
