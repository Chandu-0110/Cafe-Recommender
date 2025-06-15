import numpy as np
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
reviews = pd.read_csv('C:/Users/hp/Cafe Recommendation System/reviews.csv')
data=reviews.copy()

#data cleansing
reviews.duplicated().sum()
reviews.drop_duplicates(inplace=True)

reviews.isnull().sum()
reviews.dropna(how='any', inplace=True)
reviews = reviews.rename(columns={'Rate for two':'cost','Overall_Rating':'rating'})

reviews['cost'] = pd.to_numeric(reviews['cost'], errors='coerce')
default_value = 0  # You can choose a different default value
reviews['cost'].fillna(default_value, inplace=True)
reviews['cost'] = reviews['cost'].astype(str) #Changing the cost to string
reviews['cost'] = reviews['cost'].apply(lambda x: x.replace(',','')) #Using lambda function to replace ',' from cost
reviews['cost']=reviews['cost'].astype(float)
reviews['rating'] = pd.to_numeric(reviews['rating'], errors='coerce')

# Drop rows with missing values (NaN)
reviews = reviews.dropna()

scaler = MinMaxScaler()
reviews[['rating', 'cost']] = scaler.fit_transform(reviews[['rating', 'cost']])

feature_columns = ['rating', 'cost']
vectors = reviews[feature_columns].values

cosine_sim_mat=cosine_similarity(vectors,vectors)

target_cafe_index = 0
similar_cafes_indices = sorted(range(len(cosine_sim_mat[target_cafe_index])), key=lambda k: cosine_sim_mat[target_cafe_index][k], reverse=True)[1:]
recommended_cafes = reviews.iloc[similar_cafes_indices]

new_dataset=recommended_cafes[['Name','cost','rating']].copy()
new_dataset.duplicated().sum()

merged_df = pd.merge(new_dataset, data, left_index=True, right_index=True)
merged_df.drop(columns=['Name_y'], inplace=True)
merged_df.rename(columns={'Name_x':'Name'},inplace=True)

final_data=merged_df[['Name','Overall_Rating','Cuisine','Rate for two','City']].copy()
final_data.drop_duplicates(inplace=True)

def cost_cafes(min_price, max_price):
    final_data['Rate for two'] = final_data['Rate for two'].str.replace(',', '').astype(float)
    cafes_in_price_range = final_data[
        (final_data['Rate for two'] >= min_price) &
        (final_data['Rate for two'] <= max_price)
    ]
    return cafes_in_price_range


# Streamlit app
def main():
    st.set_page_config(
        page_title="Cafe Recommendations",
        page_icon="☕",
        layout="wide",
        initial_sidebar_state="expanded"
    )



    st.title('Cafe Recommendation System')

    st.sidebar.image("sidebar_icon.jpg", use_column_width=True)

    st.sidebar.markdown("---")
    # Sidebar for user input
    city_name = st.sidebar.text_input('Enter City Name:', '')

    # Price range input
    min_price = st.sidebar.number_input('Minimum Price for Two:', 0.0, 1.7976931348623157e308)
    max_price = st.sidebar.number_input('Maximum Price for Two:', min_price, 1.7976931348623157e308 )

    st.sidebar.markdown("---")
    t="Under the Guidance of"
    st.sidebar.markdown(t)
    st.sidebar.subheader("Ms. K N D Saile")
    pos="Assistant Professor\nDepartment of AIML&IoT"
    st.sidebar.markdown(pos)

    # Display cafes for the selected city and price range
    cafes_in_city_and_price_range = cost_cafes(min_price, max_price)
    cafes_in_city = cafes_in_city_and_price_range[cafes_in_city_and_price_range['City'].str.lower() == city_name.lower()]

    if not cafes_in_city.empty:
        st.write(f'Cafes in {city_name} with Price Range {min_price}-{max_price}')
        st.subheader("Enjoy your cafe adventure! ☕🌟")
        st.write("---")
        for index, row in cafes_in_city.iterrows():
            with st.container():
                st.header(row['Name'])
                st.subheader(f"Rating: {row['Overall_Rating']}")
                st.write(f"Cuisine: {row['Cuisine']}")
                st.write(f"Rate for two: {row['Rate for two']}")
                st.write(f"City: {row['City']}")
                st.write("---")
    else:
        st.warning(f'No cafes found in {city_name} within the specified price range')

if __name__ == '__main__':
    main()
