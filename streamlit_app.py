import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and features
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
hotel_features_with_clusters = pd.read_csv('hotel_features_with_clusters.csv')

# Load original hotel_data to get full hotel details for recommendations
hotel_data = pd.read_csv('hotels.csv') # Assuming hotels.csv is available in the environment

st.set_page_config(layout="wide")
st.title('Hotel Recommendation System')

# Sidebar for user preferences
st.sidebar.header('Your Preferences')

desired_days = st.sidebar.slider('Number of Days', min_value=1, max_value=4, value=2)
desired_price = st.sidebar.slider('Desired Price (per night)', min_value=float(hotel_data['price'].min()), max_value=float(hotel_data['price'].max()), value=float(hotel_data['price'].median()))
preferred_name = st.sidebar.selectbox('Preferred Hotel Name', options=['Any'] + list(hotel_data['name'].unique()))
preferred_place = st.sidebar.selectbox('Preferred Destination', options=['Any'] + list(hotel_data['place'].unique()))

# Preprocess user input function
def preprocess_user_input(days, price, name, place):
    # Create a DataFrame for user input
    user_input_df = pd.DataFrame([{
        'days': days,
        'price': price,
        'name': name,
        'place': place
    }])

    # Apply one-hot encoding for 'name' and 'place'
    # Get all possible columns from the training features, excluding 'cluster_label'
    training_cols_no_cluster = [col for col in hotel_features_with_clusters.columns if col != 'cluster_label']

    # One-hot encode new input, ensuring all training columns are present
    # Create dummy columns for 'name' and 'place' from user input
    user_input_encoded = pd.get_dummies(user_input_df, columns=['name', 'place'], prefix=['name', 'place'])

    # Align columns with training data features (excluding original days, price)
    # First, separate 'days' and 'price' from the expected features
    feature_cols_excluding_num = [col for col in training_cols_no_cluster if col not in ['days', 'price']]

    # Create a DataFrame with all expected one-hot encoded columns, filled with False
    aligned_input = pd.DataFrame(False, index=[0], columns=feature_cols_excluding_num)

    # Fill in the True values for the user's preferred name and place
    for col in user_input_encoded.columns:
        if col in aligned_input.columns:
            aligned_input[col] = user_input_encoded[col].iloc[0]

    # Re-add 'days' and 'price' and ensure correct order
    final_input_df = aligned_input.copy()
    final_input_df.insert(0, 'price', user_input_df['price'].iloc[0])
    final_input_df.insert(0, 'days', user_input_df['days'].iloc[0])

    # Scale numerical features
    final_input_df[['days', 'price']] = scaler.transform(final_input_df[['days', 'price']])

    # Ensure the order of columns matches the training data
    final_input_df = final_input_df.reindex(columns=[col for col in training_cols_no_cluster if col != 'cluster_label'], fill_value=0)

    return final_input_df


if st.sidebar.button('Find Recommendations'):
    # Preprocess user input
    user_preprocessed_input = preprocess_user_input(desired_days, desired_price, preferred_name, preferred_place)

    # Predict cluster for user input
    user_cluster = kmeans.predict(user_preprocessed_input)[0]

    st.subheader(f'Recommended Hotels in Cluster {user_cluster + 1}')

    # Filter original hotel_data for recommendations in the predicted cluster
    # We need to map back to original hotel_data using travelCode or other identifiers if present
    # For simplicity, we'll join based on an assumed 'hotel_id' or 'travelCode' if available
    # Since hotel_features_with_clusters has no direct link to original rows except index,
    # and original hotel_data also has no unique ID per hotel, we'll make a simplification.

    # Add cluster_label to original hotel_data for easier filtering
    if 'cluster_label' not in hotel_data.columns:
        hotel_data_for_display = hotel_data.copy()
        hotel_data_for_display = pd.merge(hotel_data_for_display.reset_index(), hotel_features_with_clusters[['cluster_label']], left_index=True, right_index=True)
        hotel_data_for_display = hotel_data_for_display.set_index('index')
    else:
        hotel_data_for_display = hotel_data # If cluster_label was already added

    # Ensure 'cluster_label' column is present in hotel_data for filtering
    # This part assumes hotel_data and hotel_features_with_clusters align by index or can be mapped
    # For a robust solution, hotel_features_with_clusters should contain original identifiers.
    # As a workaround, we'll merge on `hotel_features_with_clusters` index with `hotel_data` index

    # Let's recreate hotel_data_with_clusters more robustly if not already done
    original_hotel_data_extended = hotel_data.copy()
    # Assuming the order of rows in hotel_features_with_clusters matches original hotel_data
    # This is a critical assumption. In production, ensure a unique ID is carried through.
    original_hotel_data_extended['cluster_label'] = hotel_features_with_clusters['cluster_label']


    recommended_hotels = original_hotel_data_extended[original_hotel_data_extended['cluster_label'] == user_cluster]

    if preferred_name != 'Any':
        recommended_hotels = recommended_hotels[recommended_hotels['name'] == preferred_name]
    if preferred_place != 'Any':
        recommended_hotels = recommended_hotels[recommended_hotels['place'] == preferred_place]


    if not recommended_hotels.empty:
        # Display top 5 unique recommendations
        st.dataframe(recommended_hotels[['name', 'place', 'days', 'price']].drop_duplicates().head(5))
    else:
        st.write('No hotels found matching your criteria in the recommended cluster.')

# Provide the command to run the Streamlit app
st.markdown('---')
st.markdown('To run this Streamlit app locally, save the code above as `streamlit_app.py` and then execute in your terminal:')
st.code('streamlit run streamlit_app.py')
