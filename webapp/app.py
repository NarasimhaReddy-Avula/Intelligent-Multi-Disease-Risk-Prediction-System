"""
Streamlit Web Application
Multi-Disease AI Healthcare Platform
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Multi-Disease AI Healthcare",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🏥 Multi-Disease AI Healthcare Platform")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150?text=Logo", width=150)
    st.title("Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "📊 Risk Assessment", "📈 Visualizations", "📝 Reports", "ℹ️ About"]
    )

# Main content based on selection
if page == "🏠 Home":
    st.header("Welcome to the Multi-Disease AI Healthcare Platform")
    
    st.markdown("""
    This platform provides AI-powered risk assessment for four major diseases:
    
    | Disease | Description |
    |---------|-------------|
    | 🫀 **Heart Disease** | Cardiovascular risk assessment |
    | 🩸 **Diabetes (Type 2)** | Metabolic disorder prediction |
    | 🎗️ **Breast Cancer** | Tumor malignancy classification |
    | 🫁 **Liver Disease** | Hepatic function analysis |
    
    ### How it Works
    1. **Input your clinical data** - Enter lab results and health metrics
    2. **Get risk predictions** - AI models analyze your data
    3. **Understand the results** - SHAP + LLM explanations
    4. **Generate reports** - Download PDF reports
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Heart Disease", "Coming Soon", "AUC: TBD")
    with col2:
        st.metric("Diabetes", "Coming Soon", "AUC: TBD")
    with col3:
        st.metric("Breast Cancer", "Coming Soon", "AUC: TBD")
    with col4:
        st.metric("Liver Disease", "Coming Soon", "AUC: TBD")

elif page == "📊 Risk Assessment":
    st.header("📊 Disease Risk Assessment")
    st.info("🚧 This feature is under development. Check back after Week 11!")
    
    # Placeholder form
    with st.form("patient_form"):
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, value=50)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        
        with col2:
            blood_pressure = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
            cholesterol = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=200)
            glucose = st.number_input("Fasting Glucose", min_value=50, max_value=300, value=100)
        
        submitted = st.form_submit_button("🔮 Predict Risk")
        
        if submitted:
            st.warning("Model not yet trained. Predictions will be available after Week 9.")

elif page == "📈 Visualizations":
    st.header("📈 Model Visualizations")
    st.info("🚧 SHAP visualizations will be available after Week 9!")

elif page == "📝 Reports":
    st.header("📝 Patient Reports")
    st.info("🚧 PDF report generation will be available after Week 11!")

elif page == "ℹ️ About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### Bachelor Thesis Project (BTP) 2025
    **Institution:** IIIT Sri City  
    **Program:** B.Tech Computer Science & Engineering  
    **Supervisor:** Dr. R. Selvi
    
    ### Key Innovation
    First undergraduate multi-disease AI system combining:
    - Classical Machine Learning (XGBoost, LightGBM)
    - Deep Learning (FT-Transformer)
    - Clinical NLP (BioMistral/PubMedBERT)
    - LLM-enhanced Explainability (GPT-4/Llama)
    
    ### Technology Stack
    - **ML/DL:** PyTorch, scikit-learn, XGBoost
    - **NLP:** Transformers, PubMedBERT
    - **XAI:** SHAP, LLM explanations
    - **Backend:** FastAPI
    - **Frontend:** Streamlit
    
    ### Contact
    - GitHub: [Repository Link]
    - Email: [Your Email]
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>BTP 2025 | IIIT Sri City | Dr. R. Selvi</p>",
    unsafe_allow_html=True
)
