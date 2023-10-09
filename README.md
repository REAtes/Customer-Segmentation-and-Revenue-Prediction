# Customer Segmentation and Revenue Prediction

## Project Overview
This project aims to perform customer segmentation and revenue prediction for a gaming company based on customer attributes. The company wants to create persona-based customer definitions and segment customers based on these personas to estimate how much potential customers can generate in revenue.

## Dataset
The dataset used for this project is "persona.csv," which contains records of customer transactions with the following columns:
- `Price`: The amount spent by the customer.
- `Source`: The type of device used by the customer.
- `Sex`: The gender of the customer.
- `Country`: The country of the customer.
- `Age`: The age of the customer.

## Task

### Exploratory Data Analysis (EDA)
1. Loaded the dataset and displayed general information about it.
2. Determined the number of unique sources and their frequencies.
3. Calculate the number of unique prices and display their frequencies.
4. Counted the occurrences of each price point.
5. Counted the number of sales from each country.
6. Calculate the total revenue from sales in each country.
7. Grouped sales counts by source.
8. Calculate the average price for each country.
9. Calculate the average price for each source.

### Customer Segmentation
Perform customer segmentation based on the following attributes: `COUNTRY`, `SOURCE`, `SEX`, and `AGE`. The resulting DataFrame (`agg_df`) includes the mean price for each unique combination of these attributes.

### Sorting the Data
Sort the `agg_df` DataFrame based on the `PRICE` column in descending order.

### Renaming Index Names
Rename the index names of the DataFrame to variable names.

### Converting AGE to Categorical Variable
Convert the numeric `AGE` variable into a categorical variable with user-defined age categories (`AGECAT`) and add it to the `agg_df` DataFrame.

### Creating Customer-Level-Based Categories
Create a new variable called `customers_level_based` that combines customer attributes for segmentation. Duplicate values were removed.

### Segmenting New Customers
Segment new customers based on their predicted revenue into four segments: 'A', 'B', 'C', and 'D', using quartiles of the `PRICE` column.

### Predicting Revenue for New Customers
Implement a function `mean_revenue_prediction` to predict the revenue and segment for new customers based on their attributes. Users can input their age, source, country, and sex, and the function will provide the segment and estimated revenue.

## How to Use
1. Clone this repository to your local machine.
2. Ensure you have Python and the required libraries (NumPy, Pandas, Seaborn, Matplotlib) installed.
3. Run the Jupyter Notebook or Python script to execute the code.
4. To predict revenue for new customers, use the `mean_revenue_prediction` function by passing the customer's age, source, country, and sex as arguments.

Enjoy exploring customer segmentation and revenue prediction with this project!
