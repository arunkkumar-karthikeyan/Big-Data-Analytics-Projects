#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(dplyr)
library(DT)
library(shinydashboard)
library(fresh)
library(ggplot2)
library(plotly)
options(warn = - 1)

#__READING DATA__
datamart <- read.csv("Datamart_Team2.csv", header = TRUE, sep = ",")
countrysort <- sort(unique(datamart$Country.Name))

#__USING WHITE BACKGROUND IN DASHBOARD BODY__
mytheme <- create_theme(
  adminlte_global(
    content_bg = "#FFF",
    box_bg = "#D8DEE9", 
    info_box_bg = "#D8DEE9"
  )
)

#__INITIATING UI__
ui <- dashboardPage(
  
  #__DEFINING A DASHBOARD TITLE__
  dashboardHeader(title ="Marketing Datamart Analysis"),
  
  #__DEFINING DASHBOARD SIDEBAR MENU__
  dashboardSidebar(
    sidebarMenu(
      menuItem("Summary", tabName = "Sum", icon=icon("fas fa-solid fa-chart-simple")),
      menuItem("Demograhics", tabName = "Age", icon=icon("fas fa-solid fa-cake-candles")),
      menuItem("Products", tabName = "Prod", icon=icon("fas fa-sharp fa-solid fa-money-bill-trend-up")),
      menuItem("Customer Analysis", tabName = "CA", icon=icon("fas fa-sharp fa-solid fa-magnifying-glass-chart"))
    )
  ),
  
  #__DEFINING DASHBOARD BODY__
  dashboardBody(
    use_theme(mytheme),
    
    tabItems(
      
      #__DEFINING FIRST TAB BODY__
      tabItem(tabName = "Sum",
              h2 ("Summary Statistics"),
              box(selectInput(inputId = "cty", label = "Select your Country", choices = countrysort)),
              DT::dataTableOutput("table"),
              br(),br(),br(),br(),br(),
              box(title = "Legend", background = "light-blue", solidHeader = TRUE, 
              strong("Gold"),(" - Top Quartile Customers in terms of Spendings"),
              br(),
              strong("Silver"),(" - Customers with Median Spendings"),
              br(),
              strong("Bronze"),(" - Last Quartile Customers in terms of Spendings"),
              hr(),
              strong("Loyalists"),(" - Champion Customers"),
              br(),
              strong("Potential Loyalists"),(" - Potential Champion Customers"),
              br(),
              strong("Promising"),(" - In Decline Customers"),
              br(),
              strong("Hesitant"),(" - Passive Customers"),
              br(),
              strong("Need attention"),(" - Potential Detractors"),
              br(),
              strong("Detractors"),(" - Disappointed Customers"),
              br())),
      
      #__DEFINING SECOND TAB BODY__
      tabItem(tabName = "Age",
              h2 ("Demographics"),
              column(width = 12, box(selectInput(inputId = "country", label = "Select your Country", choices = countrysort),
                  title = "Description", background = "light-blue", solidHeader = TRUE, textOutput(outputId="ctry"))),
              tabPanel("Age",
                column(width = 6,plotOutput(outputId = "gender_graph")),
                column(width = 6,plotOutput(outputId = "lang_graph")))),
      
      #__DEFINING THIRD TAB BODY__
      tabItem(tabName = "Prod",
              h2 ("Betting, Winning and Profits per Product"),
              box(plotlyOutput(outputId = "product_bet"), 
                  plotlyOutput(outputId = "product_win")),
              box(plotlyOutput(outputId = "product_profit")),
              box(selectInput(inputId = "ct", label = "Select your Country", choices = countrysort),
                  radioButtons(inputId = "product", label   = "Select Product for country", choices = c( "Sports book fixed odd", "Sports book live action", "Casino BossMedia", "Supertoto", "Games VS", "Games Bwin", "Casino Chartwell", "Poker BossMedia"),
                               inline = TRUE),
                  sliderInput(inputId = "SportsFreq", label = "Select Frequency of Visit", min = 0, max = 500, value = c(1,100)),
                  sliderInput(inputId = "PokerFreq", label = "Select No.of Transactions", min = 0, max = 50, value = c(1,5)))),
      #__DEFINING FOURTH TAB BODY__
    tabItem(tabName = "CA",
            h2 ("Marketing Insights of Users"),
            box(plotOutput(outputId = "country_mi")),
            box(plotOutput(outputId = "sports")),
            box(radioButtons(inputId = "Segment", label = "Select type of User Segment", choices = c("Potential Loyalists", "Promising", "Need attention"), inline = TRUE)),
            box(selectInput(inputId = "countryName", label = "Country", choices = countrysort),
                sliderInput(inputId = "TotalBets", label = "No of Bets", min = 0, max = 500,value = c(1, 500))))
              )     
         )
)
#__INITIATING SERVER__
server <- function(input, output, session) {
  output$table <- DT::renderDataTable({
  #__SUBSETTING THE DATAFRAME__
  data_summ <- datamart %>% filter(Country.Name == input$cty) %>% group_by(Gender) %>% summarise("Number of Players" = n(),
                                                                                                 "Profits in EUR" = sum(ProfitabilityinEUR),
                                                                                                 "Average Revenue in EUR" = sum(ARPU),
                                                                                                 "Customer Life time value" = sum(Cust_LTV),
                                                                                                 "Total Transactions made" = sum(Tot_transactions),
                                                                                                 "Most Occuring Customer tier" = names(which.max(table(Cust_tier))),
                                                                                                 "Most Spoken Language" = names(which.max(table(Language.Description))),
                                                                                                 "Most Occuring Customer Segment" = names(which.max(table(Segment))),
                                                                                                 "Most used Application" = names(which.max(table(Application.Description))),
                                                                                                 "Most Played Game" = names(which.max(table(freq_of_visit_Sports.book.fixed.odd, freq_of_visit_Sports.book.live.action))),
                                                                                                 "Poker Transactions Morning" = sum(morning_trans_PokerBossMedia),
                                                                                                 "Poker Transactions Evening" = sum(evening_trans_PokerBossMedia),
                                                                                                 "Poker Transactions Night" = sum(night_trans_PokerBossMedia))

  DT::datatable(data_summ, rownames = FALSE, caption = "Data Table")
  })
  
  by_gender <- datamart %>% group_by(Country.Name) %>% count(Gender)
  
  output$ctry <-renderText({paste(input$country,"has",
                                   by_gender[by_gender$Country.Name == input$country & by_gender$Gender == "Male","n"],
                                   " men and ", 
                                   by_gender[by_gender$Country.Name == input$country & by_gender$Gender == "Female","n"],
                                   " women who use online gambling sites.")})
  
  #__DEFINING PLOT FOR GENDER AND LANGUAGE__
  output$gender_graph <- renderPlot(datamart %>% filter(Country.Name == input$country) %>% ggplot(aes(Gender,fill=Gender))+  geom_bar(stat='count') + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())  + theme(plot.title = element_text(hjust = 0.5, size = 15)) + theme(axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 12), axis.title = element_text(size = 15)) + labs(title = "Gender Distribution", x = "Gender", y = "Count"))    
  output$lang_graph <- renderPlot(datamart %>% filter(Country.Name == input$country) %>% ggplot(aes(Language.Description, fill=Language.Description))+  geom_bar(stat='count') + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + theme(plot.title = element_text(hjust = 0.5, size = 15)) + theme(axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 12), axis.title = element_text(size = 15)) + labs(title = "Languages Spoken", x = "Languages", y = "Count"))
  
  #__DEFINING PLOT FOR BETTING TRANSACTIONS PER PRODUCT__
  output$product_bet <- renderPlotly({
    datamart_filter <- datamart %>% filter(Country.Name == input$ct)
    if (input$product == "Sports book fixed odd"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Sports.book.fixed.odd >= input$SportsFreq[1] & freq_of_visit_Sports.book.fixed.odd <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Sports.book.fixed.odd, TotalBettingAmt_Sports.book.fixed.odd, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Betting Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    } else if (input$product == "Sports book live action"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Sports.book.live.action >= input$SportsFreq[1] & freq_of_visit_Sports.book.live.action <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Sports.book.live.action, TotalBettingAmt_Sports.book.live.action, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Betting Amount", x = "Frequency of visit", y = "Money Spent in EUR") 
    }else if (input$product == "Casino BossMedia"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Casino.BossMedia >= input$SportsFreq[1] & freq_of_visit_Casino.BossMedia <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Casino.BossMedia, TotalBettingAmt_Casino.BossMedia, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Betting Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Supertoto"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Supertoto >= input$SportsFreq[1] & freq_of_visit_Supertoto <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Supertoto, TotalBettingAmt_Supertoto, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Betting Amount", x = "Frequency of visit", y = "Money Spent in EUR") 
    }else if (input$product == "Games VS"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Games.VS >= input$SportsFreq[1] & freq_of_visit_Games.VS <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Games.VS, TotalBettingAmt_Games.VS, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Betting Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Games Bwin"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Games.bwin >= input$SportsFreq[1] & freq_of_visit_Games.bwin <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Games.bwin, TotalBettingAmt_Games.bwin, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Betting Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Casino Chartwell"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Casino.Chartwell >= input$SportsFreq[1] & freq_of_visit_Casino.Chartwell <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Casino.Chartwell, TotalBettingAmt_Casino.Chartwell, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Betting Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Poker BossMedia"){
      datamart_filter <- subset(datamart_filter, No_of_Transations_Buy_PokerBossMedia >= input$PokerFreq[1] & No_of_Transations_Buy_PokerBossMedia <= input$PokerFreq[2])
      ggplot(datamart_filter, aes(No_of_Transations_Buy_PokerBossMedia, TotTransAmt_Buy_PokerBossMedia, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Poker Buy", x = "No. of Transactions", y = "Transaction Amount")
    }
  })
  
  #__DEFINING PLOT FOR WINNING TRANSACTIONS PER PRODUCT__
  output$product_win <- renderPlotly({
    datamart_filter <- datamart %>% filter(Country.Name == input$ct)
    if (input$product == "Sports book fixed odd"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Sports.book.fixed.odd >= input$SportsFreq[1] & freq_of_visit_Sports.book.fixed.odd <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Sports.book.fixed.odd, TotWinningAmt_Sports.book.fixed.odd, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Winning Amount", x = "Frequency of visit", y = "Money Won in EUR")
    } else if (input$product == "Sports book live action"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Sports.book.live.action >= input$SportsFreq[1] & freq_of_visit_Sports.book.live.action <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Sports.book.live.action, TotWinningAmt_Sports.book.live.action, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Winning Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Casino BossMedia"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Casino.BossMedia >= input$SportsFreq[1] & freq_of_visit_Casino.BossMedia <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Casino.BossMedia, TotWinningAmt_Casino.BossMedia, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Winning Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Supertoto"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Supertoto >= input$SportsFreq[1] & freq_of_visit_Supertoto <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Supertoto, TotWinningAmt_Supertoto, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Winning Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Games VS"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Games.VS >= input$SportsFreq[1] & freq_of_visit_Games.VS <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Games.VS, TotWinningAmt_Games.VS, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Winning Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Games Bwin"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Games.bwin >= input$SportsFreq[1] & freq_of_visit_Games.bwin <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Games.bwin, TotWinningAmt_Games.bwin, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Winning Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Casino Chartwell"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Casino.Chartwell >= input$SportsFreq[1] & freq_of_visit_Casino.Chartwell <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Casino.Chartwell, TotWinningAmt_Casino.Chartwell, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Total Winning Amount", x = "Frequency of visit", y = "Money Spent in EUR")
    }else if (input$product == "Poker BossMedia"){
      datamart_filter <- subset(datamart_filter, No_of_Transations_Buy_PokerBossMedia >= input$PokerFreq[1] & No_of_Transations_Buy_PokerBossMedia <= input$PokerFreq[2])
      ggplot(datamart_filter, aes(No_of_Transations_Buy_PokerBossMedia, TotTransAmt_Sell_PokerBossMedia, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Poker Buy", x = "No. of Transactions", y = "Transaction Amount")
    } 
    
  })
  
  #__DEFINING PLOT FOR PROFITS PER PRODUCT__
  output$product_profit <- renderPlotly({
    datamart_filter <- datamart %>% filter(Country.Name == input$ct)
    if (input$product == "Sports book fixed odd"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Sports.book.fixed.odd >= input$SportsFreq[1] & freq_of_visit_Sports.book.fixed.odd <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Sports.book.fixed.odd, Profitability_Sports.book.fixed.odd, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Profits", x = "Frequency of visit", y = "Profits in EUR")
    } else if (input$product == "Sports book live action"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Sports.book.live.action >= input$SportsFreq[1] & freq_of_visit_Sports.book.live.action <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Sports.book.live.action, Profitability_Sports.book.live.action, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Profits", x = "Frequency of visit", y = "Profits in EUR")
    }else if (input$product == "Casino BossMedia"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Casino.BossMedia >= input$SportsFreq[1] & freq_of_visit_Casino.BossMedia <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Casino.BossMedia, Profitability_Casino.BossMedia, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Profits", x = "Frequency of visit", y = "Profits in EUR")
    }else if (input$product == "Supertoto"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Supertoto >= input$SportsFreq[1] & freq_of_visit_Supertoto <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Supertoto, Profitability_Supertoto, fill = Gender)) + geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Profits", x = "Frequency of visit", y = "Profits in EUR")
    }else if (input$product == "Games VS"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Games.VS >= input$SportsFreq[1] & freq_of_visit_Games.VS <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Games.VS, Profitability_Games.VS, fill = Gender)) + geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Profits", x = "Frequency of visit", y = "Profits in EUR")
    }else if (input$product == "Games Bwin"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Games.bwin >= input$SportsFreq[1] & freq_of_visit_Games.bwin <= input$SportsFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Games.bwin, Profitability_Games.bwin, fill = Gender))+ geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Profits", x = "Frequency of visit", y = "Profits in EUR")
    }else if (input$product == "Casino Chartwell"){
      datamart_filter <- subset(datamart_filter, freq_of_visit_Casino.Chartwell >= input$PokerFreq[1] & freq_of_visit_Casino.Chartwell <= input$PokerFreq[2])
      ggplot(datamart_filter, aes(freq_of_visit_Casino.Chartwell, Profitability_Casino.Chartwell, fill = Gender)) + geom_col(position = "dodge") + theme_bw() + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + labs(title = "Profits", x = "Frequency of visit", y = "Profits in EUR")
    }
  })
  #__DEFINING PLOT FOR CUSTOMER MARKETING ANALYSIS__
  
  output$country_mi <- renderPlot({
    datamart_count <- datamart %>% group_by(Segment, Country.Name) %>% summarise(count = n()) %>% top_n(n = 10, wt = count)
  if (input$Segment == "Potential Loyalists"){
    datamart_count <- subset(datamart_count, Segment == "Potential Loyalists")
    ggplot(datamart_count, aes(x = Country.Name, y=count, fill = Country.Name)) +  geom_bar(stat = "identity") + theme_bw() + theme(plot.title = element_text(hjust = 0.5, size=15)) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + theme(axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 12), axis.title = element_text(size = 15)) + labs(title = "Countrywise", x = "Countries", y = "Number of Users")
  }else if (input$Segment == "Promising"){
    datamart_count <- subset(datamart_count, Segment == "Promising")
    ggplot(datamart_count, aes(x = Country.Name, y=count, fill = Country.Name)) +  geom_bar(stat = "identity") + theme_bw() + theme(plot.title = element_text(hjust = 0.5, size=15)) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + theme(axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 12), axis.title = element_text(size = 15)) + labs(title = "Countrywise", x = "Countries", y = "Number of Users")
  }else if (input$Segment == "Need attention"){
    datamart_count <- subset(datamart_count, Segment == "Need attention")
    ggplot(datamart_count, aes(x = Country.Name, y=count, fill = Country.Name)) +  geom_bar(stat = "identity") + theme_bw() + theme(plot.title = element_text(hjust = 0.5, size=15)) + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank()) + theme(axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 12), axis.title = element_text(size = 15)) + labs(title = "Countrywise", x = "Countries", y = "Number of Users")
    }
    })
  #__DEFINING PLOT FOR BETS VERSUS PROFIT__
  
  output$sports <- renderPlot({
    datamart_sports <- subset(datamart, Total_No_of_Bets >= input$TotalBets[1] & Total_No_of_Bets <= input$TotalBets[2] & datamart$Country.Name == input$countryName)
    ggplot(datamart_sports, aes(ProfitabilityinEUR, Total_No_of_Bets)) + geom_point() + geom_rug(col="steelblue",alpha=0.1, size=.5)
    })
}

#__RUN THE SHINY APPLICATION__
shinyApp(ui = ui, server = server)
