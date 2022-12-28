#___________________GROUP ASSIGNMENT SUBMISSION - FINAL________________________#
#___________________BUSINESS ANALYTICAL TOOLS - OPEN SOURCE____________________#
#___________________________________TEAM 02____________________________________#
if(!require("lubridate")) install.packages("lubridate")
if(!require("haven")) install.packages("haven")
if(!require("dplyr")) install.packages("dplyr")
if(!require("tidyverse")) install.packages("tidyverse")
if(!require("readxl")) install.packages("readxl")
if(!require("rfm")) install.packages("rfm")
if(!require("ggplot2")) install.packages("ggplot2")
if(!require("gridExtra")) install.packages("gridExtra")
options(warn = - 1)


#__________________READ-IN NECESSARY DATA______________________________________#

#__READ IN DEMOGRAPHICS DATA__
demographics <- read_sas("RawDataIDemographics.sas7bdat")

#__READ IN USER AGGREGATION DATA__
daily <- read_sas("RawDataIIUserDailyAggregation.sas7bdat")

#__READ IN POKER CHIP CONVERSIONS DATA__
pokerchips <- read_sas("RawDataIIIPokerChipConversions.sas7bdat")


#________________SUMMARY OF THE READ-IN DATA____________________________________#
summary(demographics)
summary(daily)
summary(pokerchips)


#________________DIMENSIONS OF THE TABLES_______________________________________#
dim(demographics)
dim(daily)
dim(pokerchips)


#________________PREPARING DEMOGRAPHICS DATA_____________________________________#

#__CONVERTING COLUMNS WITH DATES INTO DATE FORMAT__
demographics$FirstPay <- ymd(demographics$FirstPay)
demographics$FirstSp <- ymd(demographics$FirstSp)
demographics$FirstAct <- ymd(demographics$FirstAct)
demographics$FirstCa <- ymd(demographics$FirstCa)
demographics$FirstGa <- ymd(demographics$FirstGa)
demographics$FirstPo <- ymd(demographics$FirstPo)
demographics$RegDate <- ymd(demographics$RegDate)

#__FILTERING THE OBSERVATIONS WHICH ARE OUT OF THE PERIOD FROM FEB 1 2005 TO SEP 30 2005__
demographics <- demographics %>% filter(FirstPay >= as.Date("2005-02-01") & FirstPay <= as.Date("2005-09-30"))

#__COUNTRIES, LANGUAGES AND APPICATIONS - APPENDIX TO EXCEL FILES__

#__IMPORTING THE EXCEL FILES__
country <- read_excel(path = "Countries.xlsx")
language <- read_excel(path = "Languages.xlsx")
application <- read_excel(path = "Application.xlsx")

#__LEFT MERGING THE IMPORTED FILES ONTO DEMOGRAPHICS__
demographics <- merge(demographics, country, by = "Country", all.x = TRUE)
demographics <- merge(demographics, language, by = "Language", all.x = TRUE)
demographics <- merge(demographics, application, by = "ApplicationID", all.x = TRUE)

#__CLEANING THE GENDER COLUMN__

#__CHECK FOR MISSING VALUES__
demographics[is.na(demographics$Gender),]

#__DROPPING THE MISSING VALUE__
demographics <- demographics %>% drop_na(Gender)

#__RECODING GENDER VALUES: 1 AS MALE AND 0 AS FEMALE__
demographics$Gender <- recode(demographics$Gender,"1" = "Male", "0" = "Female")

#__RECHECK FOR MISSING VALUES__
demographics[is.na(demographics$Gender),]

#__DROPPING THE UNNECESSARY COLUMNS
demographics <- demographics %>% select(-c("ApplicationID","Language","Country"))

#__CREATING NEW VARIABLES IN DEMOGRAPHIC TABLE__
#__CALCULATING THE NUMBER OF DAYS SINCE USER REGISTRATION AND FIRST INTERACTION__
demographics <- demographics %>% mutate(diff_firstAct = difftime(FirstAct, RegDate, units="days"),
                                        diff_firstPay = difftime(FirstPay, RegDate, units="days"),
                                        diff_firstsp = difftime(FirstSp, RegDate, units="days"),
                                        diff_firstca = difftime(FirstCa, RegDate, units="days"),
                                        diff_firstga = difftime(FirstGa, RegDate, units="days"),
                                        diff_firstpo = difftime(FirstPo, RegDate, units="days"))
#__HEAD OF DEMOGRAPHICS__
head(demographics)


#________________PREPARING USERAGGREGATION DATA__________________________________#

#__CONVERTING COLUMNS WITH DATES INTO DATE FORMAT__
daily$Date <- ymd(daily$Date)

#__REMOVING THE TRANSACTION OF THE CUSTOMERS WHOSE TRANSACT DATE IS EARLIER THAN FIRST PAY__
a <- merge(x = daily , y = demographics[ , c("UserID", "FirstPay")], by = "UserID",all.x = TRUE)
daily <- a %>% filter(Date >= FirstPay & Date < as.Date("2005-09-30") & Date > as.Date("2005-02-01"))
rm(a)

#__PRODUCTID - APPENDIX TO EXCEL FILES__

#__IMPORTING THE EXCEL FILE__
product <- read_excel(path = "Product.xlsx")

#__MERGING WITH USER AGGREGATION TABLE__
daily <- merge(daily, product, by = "ProductID", all.x = TRUE)

#__CONVERTING NEGATIVE VALUES IN BETS, STAKES, WINNINGS TO 0__
daily$Bets <- ifelse(daily$Bets < 0 , 0 ,daily$Bets)
daily$Stakes <- ifelse(daily$Stakes < 0, 0,daily$Stakes)
daily$Winnings <- ifelse(daily$Winnings < 0,0,daily$Winnings)

#__RENAMING THE DATE COLUMN__
rename(daily, BA_Date = Date)


#__HEAD OF USERAGGREGATION__
head(daily)

#__STATISTICAL VARIABLES OF USER AGGREGATION TABLE__
user_daily_aggregation <- daily %>% group_by(UserID, `Product Description`) %>% summarise(LatestBetDate = max(Date),
                                                                                          FirstBetDate = min(Date),
                                                                                          MaxBettingAmt = max(Stakes),
                                                                                          MinBettingAmt = min(Stakes),
                                                                                          TotalBettingAmt = sum(Stakes),
                                                                                          AvgBettingAmt = mean(Stakes),
                                                                                          MaxWinningAmt = max(Winnings),
                                                                                          MinWinningAmt = min(Winnings),
                                                                                          TotWinningAmt = sum(Winnings),
                                                                                          AvgWinningAmt = mean(Winnings),
                                                                                          TotalBets = sum(Bets),
                                                                                          AvgBets = mean(Bets),
                                                                                          freq_of_visit= n(),
                                                                                          Profitability = (TotalBettingAmt-TotWinningAmt),
                                                                                          Gain_Loss_Cust_Ratio = TotWinningAmt / TotalBettingAmt,
                                                                                          ProfitLoss = ifelse(Profitability>0, 'Profit', 'Loss'),
                                                                                          No_ActiveDays = ifelse((LatestBetDate == FirstBetDate),1,LatestBetDate - FirstBetDate))

#__PIVOTING USER DAILY AGGREGATION STATISTICAL DATA__
ColName_daily <- colnames(user_daily_aggregation)[3:ncol(user_daily_aggregation)]
user_daily_aggregation_pivot <- pivot_wider(user_daily_aggregation, names_from = `Product Description`, values_from = ColName_daily)


#__________________PREPARING POKERCHIPS DATA______________________________________#

#__EXTRACTING DATE FROM TransDateTime COLUMNS__
pokerchips$TransDateTime <- ymd_hms(pokerchips$TransDateTime)
pokerchips$TransDate <- date(pokerchips$TransDateTime)
pokerchips$TransMonth <- month(pokerchips$TransDateTime)
pokerchips$TransDay <- day(pokerchips$TransDateTime)
pokerchips$TransHour <- hour(pokerchips$TransDateTime)

#__REMOVING THE TRANSACTION OF THE CUSTOMERS WHOSE TRANSACT DATE IS EARLIER THAN FIRST PAY
#__AND IN DATE OF DEMOGRAPHIC DATASET ALSO BETWEEN THE PERIOD FEB 1,2005 TO SEP 30,2005__
b <- merge(x = pokerchips, y = demographics[ , c("UserID", "FirstPay")], by = "UserID",all.x = TRUE)
pokerchips <- b %>% filter(TransDate >= FirstPay & TransDate <= as.Date("2005-09-30") & TransDate >= as.Date("2005-02-01"))
rm(b)

#__RECODING TRANSACTION TYPE VALUES: 124 AS SELL AND 24 AS BUY__
pokerchips$TransType <- recode(pokerchips$TransType,"124" = "Sell", "24" = "Buy")

#__HEAD OF POKERCHIPS__
head(pokerchips)
a<-unique(user_daily_aggregation$UserID)

#__STATISTICAL VARIABLES FOR POKER CHIPS TABLE__
poker_chip_transaction <- pokerchips %>% group_by(UserID, `TransType`) %>% summarise(FirstTransDate = min(TransDate), 
                                                                                      LatestTransDate = max(TransDate),
                                                                                      TotTransAmt = round(sum(TransAmount),2),
                                                                                      AvgTransAmt = round(mean(TransAmount),2),
                                                                                      MaxTransAmt = round(max(TransAmount),2),
                                                                                      MinTransAmt = round(min(TransAmount),2),
                                                                                      No_of_Transations = n(),
                                                                                      No_ActiveDays = ifelse((LatestTransDate == FirstTransDate),1,LatestTransDate - FirstTransDate))

#__PIVOT WIDER POKERCHIPS DATA__
ColName_trans <- colnames(poker_chip_transaction)[3:ncol(poker_chip_transaction)]
poker_chip_transaction_pivot <- pivot_wider(poker_chip_transaction, names_from = `TransType`, values_from = ColName_trans)

#__CLASSIFICATION OF TRANSACTIONS BASED ON TIME OF DAY__

#__TRANSACTIONS DONE IN MORNING__
c <- pokerchips %>% group_by(UserID) %>% mutate(morning_pokerchip_trans = ifelse(TransHour >= 6 & TransHour < 12, 1, 0))
poker_chip_transaction_01 <- c %>% group_by(UserID) %>% summarise(morning_trans = sum(morning_pokerchip_trans))
poker_chip_transaction_pivot[,18] <- poker_chip_transaction_01 [2]
rm(poker_chip_transaction_01)

#__TRANSACTIONS DONE IN AFTERNOON__
d <- pokerchips %>% group_by(UserID) %>% mutate(evening_pokerchip_trans = ifelse(TransHour >= 12 & TransHour < 18, 1, 0))
poker_chip_transaction_01 <- d %>% group_by(UserID) %>% summarise(evening_trans = sum(evening_pokerchip_trans))
poker_chip_transaction_pivot[,19] <- poker_chip_transaction_01 [2]
rm(poker_chip_transaction_01)

#__TRANSACTIONS DONE IN NIGHT__
e <- pokerchips %>% group_by(UserID) %>% mutate(night_pokerchip_trans = ifelse(TransHour >= 18 | TransHour < 6, 1,0))
poker_chip_transaction_01 <- e %>% group_by(UserID) %>% summarise(night_trans = sum(night_pokerchip_trans))
poker_chip_transaction_pivot[,20] <- poker_chip_transaction_01 [2]
rm(c,d,e, poker_chip_transaction_01)

#__RENAMING THE POKERCHIPS DATA COLUMNS WITH SUFFIX OF PRODUCT ID 3__
poker_chip_transaction_pivot <- poker_chip_transaction_pivot %>% setNames(paste0(names(.),'_PokerBossMedia'))

#__RENAMING THE USERID COLUMN__
poker_chip_transaction_pivot <- rename(poker_chip_transaction_pivot, UserID = UserID_PokerBossMedia)


#_________________________MERGING DATA_____________________________________________#

dem_user_merge <- merge(demographics,user_daily_aggregation_pivot, by.x = "UserID", all.x = TRUE)
datamart <- merge(dem_user_merge, poker_chip_transaction_pivot, by.x = "UserID", all.x=TRUE)
summary(datamart)
colnames(datamart)


#________________CREATING ADDITIONAL NEW VARIABLES FOR DATAMART_____________________#

#__LATEST ACTIVE DATE OF THE USER__
#__USING THE COLUMNS 19 TO 25 ("LatestBetDate_Sports book fixed-odd", "LatestBetDate_Sports book live-action", "LatestBetDate_Casino Chartwell"         
#__"LatestBetDate_Games VS", "LatestBetDate_Games bwin", "LatestBetDate_Casino BossMedia", "LatestBetDate_Supertoto") TO FIND THE LAST ACTIVE DATE OF THE USER

ColName_datamart <- colnames(datamart)[19:25]
datamart[, "Last_Active_Date"] <- as.Date(apply(datamart[, c(ColName_datamart, 
                                                             "LatestTransDate_Buy_PokerBossMedia",
                                                             "LatestTransDate_Sell_PokerBossMedia")], 1, max, na.rm=TRUE),"%Y-%m-%d")

#__LENGTH OF RELATIONSHIP__
datamart[,"LOR"] <- ifelse(datamart$Last_Active_Date == datamart$FirstPay,1,datamart$Last_Active_Date - datamart$FirstPay)
datamart$LOR <- replace(datamart$LOR,is.na(datamart$LOR), 1)

#__TOTAL WINNINGS ACROSS ALL PRODUCTS
#__SELECTING COLUMNS FROM 75 TO 81 ("TotWinningAmt_Sports book fixed-odd", "TotWinningAmt_Sports book live-action"
#__"TotWinningAmt_Casino Chartwell","TotWinningAmt_Games VS", "TotWinningAmt_Games bwin", "TotWinningAmt_Casino BossMedia"
#__"TotWinningAmt_Supertoto") AND 142 (TotTransAmt_Buy_PokerBossMedia) CORRESPONDING TO TOTAL WINNINGS OF EACH PRODUCT__

datamart[,"Total_Winnings_in_EUR"] <- round(rowSums(datamart[, c(as.numeric(75:81,142))], na.rm=TRUE),2)

#__TOTAL STAKES ACROSS ALL PRODUCTS__
#__SELECTING COLUMNS FROM 47 TO 53 ("TotalBettingAmt_Sports book fixed-odd", "TotalBettingAmt_Sports book live-action"
#__"TotalBettingAmt_Casino Chartwell", "TotalBettingAmt_Games VS", "TotalBettingAmt_Games bwin", "TotalBettingAmt_Casino BossMedia"         
#__"TotalBettingAmt_Supertoto") AND 136 ("TotTransAmt_Sell_PokerBossMedia") CORRESPONDING TO TOTAL WINNINGS OF EACH PRODUCT__

datamart[,"Total_Stakes_in_EUR"] <- (round(rowSums(datamart[ , c(as.numeric(47:53,143))], na.rm=TRUE),2))

#__TOTAL BETS ACROSS ALL PRODUCTS__
#__SELECTING COLUMNS FROM 89 TO 95 ("TotalBets_Sports book fixed-odd", "TotalBets_Sports book live-action"
#__"TotalBets_Casino Chartwell", "TotalBets_Games VS", "TotalBets_Games bwin", "TotalBets_Casino BossMedia"         
#__"TotalBets_Supertoto") CORRESPONDING TO TOTAL BETS ACROSS EACH PRODUCT__

datamart[,"Total_No_of_Bets"] <- (round(rowSums(datamart[ , c(as.numeric(89:95))], na.rm=TRUE),2))

#__PROFITABILITY PER USER ACROSS ALL PRODUCTS__
datamart[,"ProfitabilityinEUR"] <- round((datamart$Total_Stakes_in_EUR - datamart$Total_Winnings_in_EUR),2)
datamart[,"Status"] <-ifelse(datamart$ProfitabilityinEUR>0, 'Profit', 'Loss')

#__CUSTOMER TIER BASED ON SPENDING__
datamart[,"Cust_tier"] <- (case_when(between(datamart$Total_Stakes_in_EUR, 805, 1127196) ~ "Gold",
                                     between(datamart$Total_Stakes_in_EUR, 226, 805) ~ "Silver",
                                     between(datamart$Total_Stakes_in_EUR, 0, 226) ~ "Bronze"))

#__TOTAL ACTIVE DAYS ACROSS ALL PRODUCTS__
datamart$maxPokerDate <- pmax(datamart$No_ActiveDays_Buy_PokerBossMedia,datamart$No_ActiveDays_Sell_PokerBossMedia,na.rm=TRUE)
datamart[,"Total_Active_Days"] <- round(rowSums(datamart[, c(131:137,165)], na.rm=TRUE),2)
datamart$Total_Active_Days <- replace(datamart$Total_Active_Days,datamart$Total_Active_Days == 0, 1)

#__CUSTOMER LIFETIME VALUE__
#______https://www.quora.com/Whats-the-average-Customer-Lifetime-Value-of-an-online-casino-client-in-your-country-or-company_________#

datamart[,"Cust_LTV"] <- round(datamart$ProfitabilityinEUR/(datamart$Total_Active_Days*datamart$LOR),4)

#__TOTAL NUMBER OF TRANSACTIONS ACROSS ALL PRODUCTS__
datamart[,"Tot_transactions"] <- round(rowSums(datamart[, c(103:109,150,151)], na.rm=TRUE),2)

#__AVERAGE REVENUE PER USER__
datamart[,"ARPU"] <- round((datamart$Total_Stakes_in_EUR/datamart$Total_Active_Days),2)

#__DEPOSIT PER BET__

datamart[,"DepositperBet"] <- round((datamart$Total_Stakes_in_EUR/datamart$Total_No_of_Bets),2)

dataFP <- daily %>% group_by(UserID) %>% summarise(FavProduct = first(`Product Description`))

dataFP<-dataFP[is.na(dataFP$FavProduct) == FALSE, ]


#_________________________MERGING DATAFP DATASET WITH DATAMART__________________________#
datamart <- merge(datamart, dataFP, by.x = "UserID", all.x=TRUE)

#__FINDING OUT THE MAXIMUM VALUE OF FREQUENCY PER USER FOR SPORTS BETTING AND POKER TRANSACTIONS__
datamart$freq <- pmax(datamart$`freq_of_visit_Sports book fixed-odd`,
                      datamart$`freq_of_visit_Sports book live-action`,
                      datamart$`freq_of_visit_Casino Chartwell`,
                      datamart$`freq_of_visit_Games VS`,
                      datamart$`freq_of_visit_Games bwin`,
                      datamart$`freq_of_visit_Casino BossMedia`,
                      datamart$freq_of_visit_Supertoto, na.rm = TRUE)
datamart$freq[is.na(datamart$freq)] = 1

#__CALCULATING RECENCY, FREQUENCY AND MONETARY VALUE__
rfm <- datamart %>% group_by(UserID) %>% summarise(recency = as.numeric(as.Date("2005-09-30") - as.Date(Last_Active_Date)) , frequency = freq,
                                                   monetary_value = round((Total_Winnings_in_EUR),0))

#__FILL NA VALUES WITH ZERO AND REMOVING FREQ FROM DATAMART__
rfm[is.na(rfm)] = 0

#__PLOTTING DENSITY PLOTS TO UNDERSTAND THE DISTRIBUTION__
r <- ggplot(rfm) +geom_density(aes(x= recency))
f <- ggplot(rfm) +geom_density(aes(x = frequency))
m <- ggplot(rfm) +geom_density(aes(x = monetary_value))
grid.arrange(r, f, m, nrow = 3)

#__CONVERTING RFM VARIABLE FROM FLOAT TO INTEGER__
rfm$UserID <- as.integer(rfm$UserID)
rfm$recency <- as.integer(rfm$recency)
rfm$monetary_value <- as.integer(rfm$monetary_value)

#__GETTING RFM SUMMARY STATISTICS__
summary(rfm)

#__CALCULATE RFM SCORE__
#______https://medium.com/analytics-vidhya/customer-segmentation-using-rfm-analysis-in-r-cd8ba4e6891_________#
rfm$Recency_score <- 0
rfm$Recency_score[rfm$recency >= 191] <- 1
rfm$Recency_score[rfm$recency <191 & rfm$recency >= 73] <- 2
rfm$Recency_score[rfm$recency <73 & rfm$recency >= 6] <- 3
rfm$Recency_score[rfm$recency < 6] <- 4

rfm$Freq_score<- 0
rfm$Freq_score[rfm$frequency >=41] <- 4
rfm$Freq_score[rfm$frequency <41 & rfm$frequency >= 18] <- 3
rfm$Freq_score[rfm$frequency <18 & rfm$frequency >= 6] <- 2
rfm$Freq_score[rfm$frequency <6] <- 1

rfm$Monetary_score <- 0
rfm$Monetary_score[rfm$monetary_value >= 698] <- 4
rfm$Monetary_score[rfm$monetary_value < 698 & rfm$monetary_value >= 170] <- 3
rfm$Monetary_score[rfm$monetary_value < 170 & rfm$monetary_value >= 34] <- 2
rfm$Monetary_score[rfm$monetary_value <34] <- 1

rfm <- rfm %>% mutate(RFM_score = Recency_score +10 * Freq_score + 100 * Monetary_score)

#__SEGMENTING CUSTOMERS BASED ON RFM SCORE__
rfm$Segment <- "0"
rfm$Segment[which(rfm$RFM_score %in% c(244, 344, 424, 434, 441, 442, 443, 444))] <-"Loyalists"
rfm$Segment[which(rfm$RFM_score %in% c(332, 333, 334, 342, 343, 412, 413, 414, 421, 422, 423, 424, 431, 432, 433))] <- "Potential Loyalists"
rfm$Segment[which(rfm$RFM_score %in% c(233, 234, 241, 311, 312, 313, 314, 321, 322, 323, 324, 331, 341))] <- "Promising"
rfm$Segment[which(rfm$RFM_score %in% c(124, 133, 134, 142, 143, 144, 214, 224, 232, 242, 243))] <- "Hesitant"
rfm$Segment[which(rfm$RFM_score %in% c(122, 123, 131, 132, 141, 212, 213, 221, 222, 223, 231))] <- "Need attention"
rfm$Segment[which(rfm$RFM_score %in% c(111, 112, 113, 114, 121, 131, 211,311, 411 ))] <- "Detractors"


#_________________________MERGING RFM DATASET WITH DATAMART__________________________#
datamart <- merge(datamart, rfm, by.x = "UserID", all.x=TRUE)

#__REPLACE ALL NA'S IN NUMERIC COLUMNS WITH ZERO__
x <- dplyr::select_if(datamart, is.numeric)
y <- dplyr::select_if(datamart, is.character)
z <- dplyr::select_if(datamart, is.Date)

x <- replace(x, is.na(x), 0)
x <- round(x,digits=2)
y$Gender <- replace(y$Gender, is.na(y$Gender),"Other")
datamart <- cbind(x,y,z)
rm(x,y,z)

#__REMOVING VARIABLES FROM DATAMART __
datamart <- datamart %>% select(-c("freq","maxPokerDate"))


#_________________________EXPORTING DATA MART AS CSV_____________________________________________#

write.csv(datamart,"Datamart_Team2.csv", row.names = FALSE)


#_________________________SAVING FINAL DATAMART__________________________________________________#

save(datamart, file="Datamart_Team2.Rdata")
