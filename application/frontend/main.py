import requests
import streamlit as st
import plotly.graph_objects as go
import json

# Setel judul halaman dan favicon
st.set_page_config(page_title="Bot or Not?", page_icon="🤖")

# Mendefinisikan header
st.title('🤖 Bot or Not?')

# Menampilkan form input
with st.form(key='user_input_form'):
    st.subheader("User Information")

    # Formulir input dengan kolom 2 columns
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input('Name', '')
        gender = st.selectbox('Gender', ['Male', 'Female'])
        email_id = st.text_input('Email Address', '')
        is_glogin = st.checkbox('Uses Google Login', value=True)
        follower_count = st.number_input('Follower Count', min_value=0, step=1)
        following_count = st.number_input('Following Count', min_value=0, step=1)
        dataset_count = st.number_input('Dataset Count', min_value=0, step=1)
        code_count = st.number_input('Notebook Count', min_value=0, step=1)

    with col2:
        discussion_count = st.number_input('Discussion Count', min_value=0, step=1)
        avg_nb_read_time_min = st.number_input('Avg Notebook Read Time (min)', min_value=0.0, step=0.1)
        registration_ipv4 = st.text_input('Registration IPv4', '')
        registration_location = st.text_input('Registration Location', '')
        total_votes_gave_nb = st.number_input('Votes on Notebooks', min_value=0, step=1)
        total_votes_gave_ds = st.number_input('Votes on Datasets', min_value=0, step=1)
        total_votes_gave_dc = st.number_input('Votes on Discussions', min_value=0, step=1)

    # Submit button
    submit_button = st.form_submit_button(label='Predict', use_container_width=True)

# Handle form submission
if submit_button:
    user_input = {
        "NAME": name,
        "GENDER": gender,
        "EMAIL_ID": email_id,
        "IS_GLOGIN": is_glogin,
        "FOLLOWER_COUNT": follower_count,
        "FOLLOWING_COUNT": following_count,
        "DATASET_COUNT": dataset_count,
        "CODE_COUNT": code_count,
        "DISCUSSION_COUNT": discussion_count,
        "AVG_NB_READ_TIME_MIN": avg_nb_read_time_min,
        "REGISTRATION_IPV4": registration_ipv4,
        "REGISTRATION_LOCATION": registration_location,
        "TOTAL_VOTES_GAVE_NB": total_votes_gave_nb,
        "TOTAL_VOTES_GAVE_DS": total_votes_gave_ds,
        "TOTAL_VOTES_GAVE_DC": total_votes_gave_dc
    }

    # Send request to FastAPI
    with st.spinner("🔍 Analyzing user data..."):
        try:
            response = requests.post('http://localhost:8000/predict/', json=user_input)

            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction')
                bot_probability = result.get('bot_probability')

                # Save to session state
                st.session_state.user_input = user_input
                st.session_state.prediction = prediction
                st.session_state.bot_probability = bot_probability
                st.session_state.name = name

            

            else:
                st.error(f"❌ Server Error: {response.status_code}")
                st.write("Detail:", response.text)

        except requests.exceptions.RequestException as e:
            st.error("🚫 Failed to connect to the prediction server.")
            st.write(f"Error Details: {str(e)}")

# Display prediction result (after prediction)
if "prediction" in st.session_state and "bot_probability" in st.session_state:
    user_input = st.session_state.user_input
    prediction = st.session_state.prediction
    bot_probability = st.session_state.bot_probability
    name = st.session_state.name

    st.subheader("📊 Prediction Result")
            
    if prediction == 1:
        st.error("⚠️ Prediction: User is likely a **BOT**!")
    else:
        st.success("✅ Prediction: User is likely a **HUMAN**!")

    # Gauge chart for bot probability
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bot_probability,
        title={'text': "Bot Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if prediction == 1 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "pink"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': bot_probability
            }
        }
    ))
    
    st.plotly_chart(fig, use_container_width=True)

    # Skor Risiko dan Kategori
    if bot_probability > 70:
        risk_level = "⚠️ **High Risk** - Strong indicators of automation."
        suggestion = "🛑 Recommended for deeper investigation."
    elif bot_probability > 30:
        risk_level = "🟡 **Moderate Risk** - Some suspicious patterns."
        suggestion = "⚠️ Monitor activity or request additional verification."
    else:
        risk_level = "🟢 **Low Risk** - Likely a genuine human user."
        suggestion = "✅ No immediate action needed."

    st.markdown(f"**Bot Risk Category:** {risk_level}")
    st.info(suggestion)

    # Expand input details
    with st.expander("📄 View Input Details"):
        st.json(user_input)

    # Format laporan sebagai dict
    report_data = {
        "Name": name,
        "Email": email_id,
        "Prediction": "Bot" if prediction == 1 else "Human",
        "Bot Probability (%)": round(bot_probability, 2),
        "Risk Category": risk_level,
        "Suggestion": suggestion,
        "Input Summary": user_input
    }

    st.download_button(
        label="📥 Download Prediction Report (JSON)",
        data=json.dumps(report_data, indent=2),
        file_name=f"{name.replace(' ', '_')}_bot_prediction_report.json",
        mime="application/json"
    )