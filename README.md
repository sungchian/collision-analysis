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
- The presence of outliers is evident, particularly in features like bed, bath, acre_lot, and house_size.
- The average property listing was priced at $886,657 with a standard deviation of over $2M indicating a large distribution range.
- There are missing values in the dataset that inaccurately represent some listings.
- The data represents a wide range of states that cover the Eastern, Mid-Atlantic, and Caribbean territories. As shown in the bar chart below, New York and New Jersey are the states that account for over 500,000 listings combined, followed by Massachusetts with approximately 175,000 listings.    
   <br>
      <img src="Images/img-01.png" width="500">
   <br>  
- The variables bed, bath, and house_size are positively correlated to price, meaning that the property price tends to increase as these features increase, as shown in the heat map below. (The redder the square, the weaker the correlation. The darker the gray or black, the stronger the correlation.)  
   <br>
      <img src="Images/img-02.png" width="500">
   <br>  
- In this dataset, New York appears to be the state with the highest average property price at over $1.4 million, whereas West Virginia appears to be the state with the lowest average property price, averaging around $62,000 per property. Additionally, the property price seems to increase in high-density urban areas such as New York City and parts of Boston, as shown in the map below.  
   <br>
      <img src="Images/img-03.png" width="500">
   <br>  
- Though New York and Massachusetts share similarities with their high average listing price, the features of these properties differ significantly. On a city-by-city comparison, the average house size in Massachusetts is much larger than that of New York. The smaller property size combined with the high price makes New York the state with the highest average price per square foot in this dataset.  
   <br>
      <img src="Images/img-04.png" width="500">
   <br>
- Georgia, West Virginia, and the Virgin Islands seem to have the highest average number of bedrooms and bathrooms per property.
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





 
