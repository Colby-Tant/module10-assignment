# Module 10 Assignment: Data Manipulation and Cleaning with Pandas
# UrbanStyle Customer Data Cleaning

import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

# Welcome message
print("=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING")
print("=" * 60)

# SIMULATED CSV CONTENT
csv_content = """customer_id,first_name,last_name,email,phone,join_date,last_purchase,total_purchases,total_spent,preferred_category,satisfaction_rating,age,city,state,loyalty_status
CS001,John,Smith,johnsmith@email.com,(555) 123-4567,2023-01-15,2023-12-01,12,"1,250.99",Menswear,4.5,35,Tampa,FL,Gold
CS002,Emily,Johnson,emily.j@email.com,555.987.6543,01/25/2023,10/15/2023,8,$875.50,Womenswear,4,28,Miami,FL,Silver
CS003,Michael,Williams,mw@email.com,(555)456-7890,2023-02-10,2023-11-20,15,"2,100.75",Footwear,5,42,Orlando,FL,Gold
CS004,JESSICA,BROWN,jess.brown@email.com,5551234567,2023-03-05,2023-12-10,6,659.25,Womenswear,3.5,31,Tampa,FL,Bronze
CS005,David,jones,djones@email.com,555-789-1234,2023-03-20,2023-09-18,4,350.00,Menswear,,45,Jacksonville,FL,Bronze
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS007,Robert,Davis,robert.davis@email.com,555.444.7777,04/30/2023,11/25/2023,7,$725.80,Footwear,4.5,38,Miami,FL,Silver
CS008,Jennifer,Garcia,jen.garcia@email.com,(555)876-5432,2023-05-15,2023-10-30,3,280.50,ACCESSORIES,3,25,Orlando,FL,Bronze
CS009,Michael,Williams,m.williams@email.com,5558889999,2023-06-01,2023-12-07,9,1100.00,Menswear,4,39,Jacksonville,FL,Silver
CS010,Emily,Johnson,emilyjohnson@email.com,555-321-6547,2023-06-15,2023-12-15,14,"1,875.25",Womenswear,4.5,27,Miami,FL,Gold
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS011,Amanda,,amanda.p@email.com,(555) 741-8529,2023-07-10,,2,180.00,womenswear,3,32,Tampa,FL,Bronze
CS012,Thomas,Wilson,thomas.w@email.com,,2023-07-25,2023-11-02,5,450.75,menswear,4,44,Orlando,FL,Bronze
CS013,Lisa,Anderson,lisa.a@email.com,555.159.7530,08/05/2023,,0,0.00,Womenswear,,30,Miami,FL,
CS014,James,Taylor,jtaylor@email.com,555-951-7530,2023-08-20,2023-10-10,11,"1,520.65",Footwear,4.5,,Jacksonville,FL,Gold
CS015,Karen,Thomas,karen.t@email.com,(555) 357-9512,2023-09-05,2023-12-12,6,685.30,Womenswear,4,36,Tampa,FL,Silver"""

customer_data_csv = StringIO(csv_content)

# TODO 1: Load and Explore the Dataset
#stores dataframe in variable raw_df
raw_df = pd.read_csv(StringIO(csv_content))
#stores initial missing value counts
initial_missing_counts = raw_df.isnull().sum()
#stores duplicate count
initial_duplicate_count = raw_df.duplicated().sum()

# TODO 2: Handle Missing Values
missing_value_report = raw_df.isnull().sum()
satisfaction_median = raw_df['satisfaction_rating'].median()
raw_df['satisfaction_rating'] = raw_df['satisfaction_rating'].fillna(satisfaction_median)

#use forward fill for last_purchase as a logic based assumption for CRM flows
date_fill_strategy = 'forward_fill'
raw_df['last_purchase'] = raw_df['last_purchase'].ffill()

#fill missing loyalty_status with "Bronze" (default for missing entries) and drop rows with critical missing names
raw_df['loyalty_status'] = raw_df['loyalty_status'].fillna('Bronze')
df_no_missing = raw_df.dropna(subset=['first_name', 'customer_id']).copy()

# TODO 3: Correct Data Types
#Datetime conversion
df_no_missing['join_date'] = pd.to_datetime(df_no_missing['join_date'], format='mixed')
df_no_missing['last_purchase'] = pd.to_datetime(df_no_missing['last_purchase'], format='mixed')

#converts total_spent to numeric
df_no_missing['total_spent'] = df_no_missing['total_spent'].replace(r'[\$,]', '', regex=True).astype(float)

#ensures other numeric fields are correct types
df_typed = df_no_missing.copy()
df_typed['total_purchases'] = pd.to_numeric(df_typed['total_purchases'])
df_typed['age'] = df_typed['age'].astype("Int64")

# TODO 4: Clean and Standardize Text Data
df_text_cleaned = df_typed.copy()
df_text_cleaned['first_name'] = df_text_cleaned['first_name'].str.title()
df_text_cleaned['last_name'] = df_text_cleaned['last_name'].str.title()
df_text_cleaned['preferred_category'] = df_text_cleaned['preferred_category'].str.capitalize()

# Standardize phone numbers to (XXX) XXX-XXXX
phone_format = '(XXX) XXX-XXXX'
def clean_phone(phone):
    if pd.isna(phone): return "N/A"
    digits = "".join(filter(str.isdigit, str(phone)))
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}" if len(digits) == 10 else phone

df_text_cleaned['phone'] = df_text_cleaned['phone'].apply(clean_phone)

# TODO 5: Remove Duplicates
duplicate_count = df_text_cleaned.duplicated(subset=['customer_id']).sum()
df_no_duplicates = df_text_cleaned.drop_duplicates(subset=['customer_id'], keep='first').copy()

# TODO 6: Add Derived Features
#Current date for recency calculation (assuming Dec 31, 2023)
reference_date = df_no_duplicates['last_purchase'].max()
df_no_duplicates['days_since_last_purchase'] = (reference_date - df_no_duplicates['last_purchase']).dt.days

df_no_duplicates['average_purchase_value'] = (df_no_duplicates['total_spent'] / 
                                              df_no_duplicates['total_purchases'])
#creates a purchase frequency category
def freq_cat(x):
    if x >= 10: return 'High'
    elif x >= 5: return 'Medium'
    else: return 'Low'

df_no_duplicates['purchase_frequency_category'] = df_no_duplicates['total_purchases'].apply(freq_cat)

# TODO 7: Clean Up the DataFrame
df_renamed = df_no_duplicates.rename(columns={
    'preferred_category': 'Category',
    'loyalty_status': 'Loyalty'
})

df_final = df_renamed.drop(columns=['email', 'phone']).copy()
df_final = df_final.sort_values(by='total_spent', ascending=False)

# TODO 8: Generate Insights
#calculates average spent by loyalty status
avg_spent_by_loyalty = df_final.groupby('Loyalty')['total_spent'].mean()
#finds the top preferred categories by total spent
category_revenue = df_final.groupby('Category')['total_spent'].sum().sort_values(ascending=False)
#calculates correlation between satisfaction rating and total spent
satisfaction_spend_corr = df_final['satisfaction_rating'].corr(df_final['total_spent'])

# TODO 9: Generate Final Report
print("\n" + "=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING REPORT")
print("=" * 60)

#reports on data quality issues found and how they were addressed
print(f"Data Quality Issues:")
print(f"- Missing Values: {initial_missing_counts.sum()} total missing entries")
print(f"- Duplicates: {initial_duplicate_count} duplicate records found")
print(f"- Data Type Issues: Incorrect types for total_spent (string) and dates (object).")

#describes changes made to standardize the dataset
print(f"\nStandardization Changes:")
print(f"- Names: Converted to proper case (Title Case)")
print(f"- Categories: Standardized to Capitalized (e.g., Menswear)")
print(f"- Phone Numbers: Formatted to {phone_format}")

#presents key business insights from the cleaned data
print(f"\nKey Business Insights:")
print(f"- Customer Base: {len(df_final)} unique customers")
print(f"- Revenue by Loyalty:\n{avg_spent_by_loyalty}")
print(f"- Top Category: {category_revenue.index[0]} with ${category_revenue.iloc[0]:.2f} revenue")

#displayes the first five rows of the clean, analysis ready dataset
print("\nFinal Cleaned Dataset (Top 5 Rows):")
print(df_final.head())