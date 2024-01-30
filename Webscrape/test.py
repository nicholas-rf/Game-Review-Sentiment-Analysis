import pandas as pd

df = pd.read_csv('C:\\Users\\nicho\\OneDrive\\Desktop\\Project\\Data\\gamesradar_dataset.csv')

# print(df['Signifier'].unique())



# need to drop steel series arctis, Iphone, otherweise good
value_counts = df['Title'].value_counts()
# Filter values that occur more than once
values_more_than_once = value_counts[value_counts > 1]
# Get the index (unique values) of values that occur more than once
values_more_than_once_list = values_more_than_once.index.tolist()

print(values_more_than_once_list)

df = df.drop_duplicates(subset='Title')  

# need to drop steel series arctis, Iphone, otherweise good
value_counts = df['Title'].value_counts()
# Filter values that occur more than once
values_more_than_once = value_counts[value_counts > 1]
values_once = value_counts[value_counts == 1]
print(len(values_once.index.to_list()))
# Get the index (unique values) of values that occur more than once
values_more_than_once_list = values_more_than_once.index.tolist()

print(values_more_than_once_list)




# # Filter the DataFrame to show rows where the specified column has values that occur more than once
# result_df = df[df["Title"].isin(values_more_than_once_list)]

# # Print the result
# print(result_df.head(3))
# print("Rows where '{}' occurs more than once:\n{}".format('Title', result_df))