import streamlit as st
import pandas as pd
from model_handler import load_model, make_prediction

def main():
    st.title("Income Prediction App")
    st.write("Enter the following information to predict income category:")

    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Numerical inputs
        age = st.number_input("Age", min_value=16, max_value=90, value=30)
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
        hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
        
        # Personal info
        sex = st.selectbox("Sex", ["Male", "Female"])
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])

    with col2:
        # Work-related info
        workclass = st.selectbox("Workclass", [
            "Private", "Self-emp-not-inc", "State-gov", "Federal-gov", 
            "Local-gov", "Self-emp-inc", "Without-pay", "Never-worked"
        ])
        
        occupation = st.selectbox("Occupation", [
            "Exec-managerial", "Sales", "Tech-support", "Craft-repair", 
            "Other-service", "Machine-op-inspct", "Adm-clerical", 
            "Transport-moving", "Handlers-cleaners", "Farming-fishing", 
            "Priv-house-serv", "Protective-serv", "Armed-Forces"
        ])
        
        # Education and family
        education = st.selectbox("Education", [
            "Bachelors", "HS-grad", "Some-college", "Masters", "Doctorate", 
            "Assoc-acdm", "Assoc-voc", "11th", "10th", "9th", "7th-8th", 
            "12th", "1st-4th", "5th-6th", "Preschool"
        ])
        
        marital_status = st.selectbox("Marital Status", [
            "Never-married", "Married-civ-spouse", "Divorced", 
            "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
        ])
        
        relationship = st.selectbox("Relationship", [
            "Not-in-family", "Husband", "Own-child", "Wife", "Unmarried", "Other-relative"
        ])

    # Button to make prediction
    if st.button("Predict Income", type="primary"):
        try:
            input_data = {
                'age': age,
                'capital.gain': capital_gain,
                'capital.loss': capital_loss,
                'hours.per.week': hours_per_week,
                'workclass': workclass,
                'education': education,
                'marital.status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'sex': sex
            }

            # Load model and make prediction
            model, scaler, feature_names = load_model()
            result, confidence = make_prediction(model, scaler, feature_names, input_data)
            
            # Display results with styling
            st.markdown("---")
            st.subheader("Prediction Results")
            
            if result == ">50K":
                st.success(f"🎯 **Predicted Income: {result}**")
                st.info(f"💡 **Confidence: {confidence:.1%}**")
            else:
                st.warning(f"🎯 **Predicted Income: {result}**")
                st.info(f"💡 **Confidence: {confidence:.1%}**")
            
            # Additional insights
            st.markdown("---")
            st.subheader("Input Summary")
            col_summary1, col_summary2 = st.columns(2)
            
            with col_summary1:
                st.write(f"**Age:** {age}")
                st.write(f"**Education:** {education}")
                st.write(f"**Work Class:** {workclass}")
                st.write(f"**Occupation:** {occupation}")
                
            with col_summary2:
                st.write(f"**Hours/Week:** {hours_per_week}")
                st.write(f"**Marital Status:** {marital_status}")
                st.write(f"**Capital Gain:** ${capital_gain:,}")
                st.write(f"**Capital Loss:** ${capital_loss:,}")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that the model files are properly loaded.")

    # Add some helpful information
    with st.expander("ℹ️ About this App"):
        st.write("""
        This app predicts whether a person's income is above or below $50,000 based on demographic and work-related factors.
        
        **Model Information:**
        - Uses machine learning trained on the Adult Census dataset
        - Considers factors like age, education, occupation, and work hours
        - Provides confidence scores for predictions
        
        **Note:** This is for demonstration purposes only and should not be used for actual financial decisions.
        """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Income Prediction App",
        page_icon="💰",
        layout="wide"
    )
    main()