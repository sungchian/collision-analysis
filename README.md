# 📌 Project Background 

California, specifically in the vibrant city of Irvine, stands out for its characteristic car-centric lifestyle, a defining element that shapes the state's identity. This report aims to go beyond mere statistics and delve into the nuanced factors contributing to car accidents. By doing so, our objective is not only to provide a meticulous examination of accident frequency but also to illuminate the underlying dynamics that influence these incidents.   

<i>In collaboration with Angel Sheu, Tracey Liu, and Lisa Rumao.</i>

Our project questions include:   
- What type of car/motorcycle was involved in the most crashes?
- Were there more car accidents than usual during the holidays?
- Were there more accidents around sunrise/sunset?
- Were DUIs more likely on certain days?

This project uses the California Traffic Collision Data [Dataset](https://www.kaggle.com/datasets/alexgude/california-traffic-collision-data-from-switrs/data) from Kaggle.    
- Collisions table: Covers details such as collision location, road conditions, weather, collision severity, type, and hit-and-run status.
- Parties table: Encompasses information on involved individuals, including their age, sex, and sobriety.
- Victims table: Focuses on specific individuals' injury details resulting from the collisions, offering a holistic perspective on the incidents.

# 🛠️ Analyzing tools
- Our research methodology seamlessly integrated Python and SQL for efficient data gathering and analysis. Leveraging the power of linear regression, we conducted correlation studies to unveil meaningful insights.
- The visualization aspect of our study was enriched using Matplotlib and Seaborn, enabling the creation of insightful heatmaps, bar charts, and pie charts.
- We utilized a comprehensive toolkit consisting of Pandas and NumPy for data manipulation, SQLite3 for database management, Folium for geographical visualizations, and Plotly Express for interactive data visualization in our analysis.

# 📊 Exploratory Data Analysis  
- This geographical insight is a valuable starting point for further investigation into the factors that contribute to higher incident rates in these areas.
- By identifying these locations, policymakers and local authorities can customize interventions to address unique challenges in each area, ultimately leading to targeted road safety improvements.
   <br>
      <img src="Images/img-01.png" width="500">
   <br>  
- Next, we analyzed the gender and race distribution of the collision parties and found that there was an even split, with 60% male and 40% female, and representation from diverse racial backgrounds including white, Hispanic, black, and Asian.  
   <br>
      <img src="Images/img-02.png" width="500">
   <br>  
- During our research, we paid particular attention to age-related trends and found that drivers between the ages of 20 and 30 have the highest collision rates. This finding is in line with the higher insurance rates charged to younger drivers, as shown in the age and fault distribution chart. The chart demonstrates that younger drivers are at fault more often, which affects insurance premiums.
   <br>
      <img src="Images/img-03.png" width="500">
   <br>
   <br>
      <img src="Images/img-04.png" width="500">
   <br>  
- After analyzing the data provided, we have noticed a decrease in collision rates in California. This trend is consistent and warrants further investigation into the factors that may have contributed to it, possible interventions that can be implemented, and the implications for road safety measures.
   <br>
      <img src="Images/img-05.png" width="500">
   <br>
- In different vehicle categories, motor or scooter has the most collisions, 300000 cases, for the past 20 years. Although these vehicles are not commonly seen on the street in America, they still have the highest amount of collisions, which means these vehicles can be really dangerous. The second and third highest collision amounts are from cars and trucks. They respectively had 170000 and 20000 collision cases for the past 20 years.
   <br>
       <img src="Images/img-05.png" width="500">
   <br>
- In those three categories, we analyzed the relationship between the brands and safety. For motors or scooters, harley-davidson, Ducati, Honda, and Suzuki have the most people killed in collisions. The highest rate was around 4.2%. On the contrary, BMW has the least people killed in collisions.
   <br>
       <img src="Images/img-05.png" width="500">
   <br>
- Look at the car category: Ford, Chevrolet, and Dodge have the most people killed in the collisions. The highest rate was 3.3%. On the other hand, Honda, Nissan, and BMW have the least number of people killed in collisions. However, considering the price and gas savings, most people usually buy a Honda or Nissan when they start working.
   <br>
       <img src="Images/img-05.png" width="500">
   <br>
- Look at the trucks. Dodge had the most people killed in the collisions, the rate was greater than 6%. Toyota and Nissan were the safest brands, with the fewest fatalities from collisions.
   <br>
       <img src="Images/img-05.png" width="500">
   <br>
- Look at the collisions at different times of the day. As mentioned, the categories of motorcycle or scooter, car, and truck have the most collisions. We can see that most collisions happen from 3 pm to 6 pm and the number peaks at 5 pm, which is the time the sun sets and people get off of work. When the sun is setting, the sunlight will become very strong. On top of that, many people will rush back home which might result in not driving carefully.
   <br>
       <img src="Images/img-05.png" width="500">
   <br>
- In the occurrences in the holiday chart, we can give some insights. Considering certain holidays (such as Halloween) have consistently shown high accident rates over the years, the government may consider intensifying traffic regulations during these dates.
   <br>
       <img src="Images/img-05.png" width="500">
   <br>
- Age vs. Total Alcohol Involved:
- There is a significant decline in the number of incidents involving alcohol as age increases. The highest number of alcohol-involved cases occurs among individuals between the ages of 20 to 29 (ages 16 to 20 have relatively lower cases), after which there's a gradual decrease as age advances.
   <br>
       <img src="Images/img-05.png" width="500">
   <br>
- Age vs. Total Collisions:
- Total collisions generally follow a downward trend as age progresses. The highest number of collisions tends to occur among younger individuals (ages 16 to 25), while older age groups (from around 30 onwards) experience fewer collisions.

   <br>
       <img src="Images/img-05.png" width="500">
   <br>
  
Highlighting these relationships helped us gain a deeper understanding of the data which offered valuable insight as we continued our analysis through machine learning models.  

# 👣 Our Approach  
Real estate analysis typically attempts to predict price, a continuous variable. However, we took a classification approach to this problem, since classification introduces a layer of interpretability and simplicity to our analysis, which can benefit business professionals and prospective buyers. By categorizing properties into pricing tiers (high and low), we aim to compare the accuracy and performance of each of the selected models.    

# 🧽 Data Cleaning  
1. We discretized the dependent variable.   
   <br>
      <img src="Images/img-06.png" width="600">
   <br>  
2. Then, we filled in the missing values with the mean and median values. Specifically, we used the median for the missing values in the acre_lot, house_size, and price columns. Additionally, rows with missing values in the city and zip_code columns were removed. Lastly, the records for Tennessee, South Carolina, and Virginia listings were removed because they contained a substantial amount of missing data.  
   <br>
      <img src="Images/img-07.png" width="600">
   <br>

This "cleaned" dataset served as our initial benchmark for subsequent machine learning experiments.  

# 🔍 Machine Learning Models  
### Random Forest Classifier    
- Without any pre-processing techniques, the results are as follows:  
   <br>
      <img src="Images/img-08.png" width="400">
   <br>  
The model predicts approximately 92% of instances correctly. The precision and recall of the model are relatively high, at 0.92, indicating a low rate of false positives and negatives. Additionally, the high F1-score implies that this model performs well. Overall, the initial benchmark for the Random Forest algorithm on this dataset demonstrates strong performance with high accuracy.  
- We attempted to enhance the model with random sampling, dummy variables for the state attribute, feature selection, binning, min-max scaling, and standardization pre-processing techniques. Random sampling reduced the dataset for additional features while maintaining a similar model accuracy. Implementing dummy variables for the state attribute did not change the accuracy. Since the number of attributes in the original dataset is not extremely large, feature selection was not useful. Binning underscored the nature of the data which decreased its accuracy. Standardization improved the previous pre-processing, but it had minimal effect on improving the accuracy.  
   <br>
      <img src="Images/img-09.png" width="400">
   <br>  
Overall, the Random Forest model that performed the best was the benchmark model (with no pre-processing). Many of the additional pre-processing techniques either worsened or had no impact relative to the original accuracy. However, a finding that was gained from the pre-processing was that price is likely to be influenced by location since adding the dummy variables for the state attribute improved the randomly sampled model.

### K-Nearest Neighbors Classifier  
- Without any pre-processing techniques, the results are as follows:  
   <br>
      <img src="Images/img-10.png" width="400">
   <br>  
This model predicts approximately 88% of instances correctly. The model's precision and recall are commendably high, at 0.87, suggesting a low frequency of false positives and negatives. The high F1-score further confirms the model's effectiveness. However, this model appears to be less accurate than the Random Forest model with a 92% overall accuracy.
- The pre-processing techniques that we employed were random sampling, an introduction of dunny variables for the states attribute, feature selection, binning, min-max scaling, and standardization. Most of the techniques were moderately effective, but they tended to distort the true nature of the data, leading to less accurate predictions. Standardization offered some improvement over the other pre-processing techniques. However, the impact it had on the overall accuracy was insignificant.     
   <br>
      <img src="Images/img-11.png" width="400">
   <br>  
Overall, the KNN model achieved optimal results when applied to the standardization of the bed, bath, acre_lot, and house_size attributes. The standardized KNN model achieved an 88.5% accuracy. a slight increase of 0.1% from the benchmark model.  

### Logistic Regression  
- Without any pre-processing techniques, the results are as follows:  
   <br>
      <img src="Images/img-12.png" width="400">
   <br>  
This model predicts approximately 70% of instances correctly. The model's precision and recall contained mixed results. As noted by the F1-score of 0.65 for the "high" class and 0.74 for the "low" class, this seems to suggest that the model is better at identifying the "low" class data points. Overall, this Logistic Regression model has room for improvement, particularly in capturing the "high" class data points.
- Similar to the Random Forest and KNN models, the pre-processing techniques that we employed were random sampling, an introduction of dunny variables for the states attribute, feature selection, binning, min-max scaling, and standardization. Implementing dummy variables for the states attribute increased the model's accuracy to 72%. Standardization and binning on the house_size attribute improved the model's accuracy to 75%.  
   <br>
      <img src="Images/img-13.png" width="400">
   <br>  
Overall, the Logistic Regression approach had better performance in two separate instances, binning on the house_size attribute and a combination of random sampling, dummy variables, and standardization. Both models achieved an accuracy of around 75%, an improvement of approximately 5% from the benchmark model.

# 🔑 Key Takeaways    
We implemented three machine learning models to predict whether a real estate listing could be classified as a "high" or "low" price listing. From our analysis, we concluded how different models reacted to a variety of pre-processing techniques. From our choice of pre-processing techniques, the best techniques seemed to involve a combination of random sampling, standardization, and dummy variables for the state attribute. The binning technique improved the Logistic Regression model significantly. However, binning also led to a significant decrease in the performance of the Random Forest and KNN models, suggesting the importance of retaining the original granularity for some features.  

For this dataset, the best model seemed to be the Random Forest algorithm with no pre-processing. This model had the highest accuracy of 92%. Property location also seemed to be the attribute that plays a significant role in price.  

# ☁️ Project Improvements  
This project was for the first machine learning class that I took, and it was also the first project where I applied machine learning algorithms. Knowing what I know now if I were to improve this project, I would focus on improving the Random Forest model using different boosting methods, such as Adaptive Boosting and Gradient Boosting.  





 
