# ### Task 1: Exploratory Data Analysis (EDA)
# 1. Loaded the dataset and displayed general information about it.
# 2. Determined the number of unique SOURCE and their frequencies.
# 3. Calculated the number of unique PRICE and displayed their frequencies.
# 4. Counted the occurrences of each PRICE point.
# 5. Counted the number of sales from each COUNTRY.
# 6. Calculated the total revenue from sales in each COUNTRY.
# 7. Grouped sales counts by SOURCE.
# 8. Calculated the average price for each COUNTRY.
# 9. Calculated the average price for each SOURCE.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("persona.csv")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

def check_df(dataframe, head=5):
    print("#################### Shape ####################")
    print(dataframe.shape)
    print("#################### Types ####################")
    print(dataframe.dtypes)
    print("#################### Num of Unique ####################")
    print(dataframe.nunique())  # "dataframe.nunique(dropna=False)" yazarsak null'larıda veriyor.
    print("#################### Value Count ####################")
    for col in dataframe.columns:
        print(dataframe.value_counts(col))
    print("#################### Head ####################")
    print(dataframe.head(head))
    print("#################### Tail ####################")
    print(dataframe.tail(head))
    print("#################### NA ####################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ####################")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=16, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken
        sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if dataframe[col_name].dtype == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)


total_rev_each_country = df.groupby("COUNTRY").agg({"PRICE": "sum"}).sort_values(by="PRICE", ascending=False)
total_tra_each_country = df["COUNTRY"].value_counts().sort_values(ascending=False)
total_sale_by_source = df["SOURCE"].value_counts().sort_values(ascending=False)

avg_price_for_each_country = df.groupby("COUNTRY").agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False)
avg_price_for_each_source = df.groupby("SOURCE").agg({"PRICE": "mean"}).sort_values(by="PRICE", ascending=False)

# ### Customer Segmentation

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])\
    .agg({"PRICE": "mean"})\
    .sort_values(by="PRICE", ascending=False)\
    .reset_index()


# ### Converting AGE to Categorical Variable

age_bins = [0, 18, 23, 30, 40, 70]
age_labels = ["0_18", "19_23", "24_30", "31_40", "41_70"]
agg_df["AGECAT"] = pd.cut(agg_df["AGE"], bins=age_bins, labels=age_labels)


# ### Creating Customer Level-Based Categories

agg_df["CUSTOMER_LEVEL_BASED"] = ['_'.join(i).upper() for i in agg_df.drop(["PRICE", "AGE"], axis=1).values]
customer_level_based = agg_df.groupby("CUSTOMER_LEVEL_BASED")\
    .agg({"PRICE": "mean"})\
    .sort_values("PRICE", ascending=False)\
    .reset_index()


# ### Segmenting New Customers

customer_level_based["SEGMENT"] = pd.qcut(customer_level_based["PRICE"], q=4, labels=["D", "C", "B", "A"])
customer_level_based.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})


# ### Predicting Revenue for New Customers


def mean_revenue_prediction(age, source, country, sex):
    for indeks, row in customer_level_based.iterrows():
        cnt, src, sx, start, end = customer_level_based["CUSTOMER_LEVEL_BASED"][indeks].split("_")
        start = int(start)
        end = int(end)
        if age >= start and age <= end and source == src and country == cnt and sex == sx:
            print(customer_level_based["SEGMENT"][indeks])
            print(customer_level_based["PRICE"][indeks])
            break


mean_revenue_prediction(36, "ANDROID", "TUR", "MALE")  # Output: Segment: D, Price: 29.0
mean_revenue_prediction(35, "IOS", "TUR", "FEMALE")  # Output: Segment: D, Price: 32.333333333333336


