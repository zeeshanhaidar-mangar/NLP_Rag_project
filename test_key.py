"""
Gemini API Key Tester
Save this as: test_key.py
Run with: streamlit run test_key.py
"""

import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="API Key Tester", page_icon="🔑")

st.title("🔑 Gemini API Key Tester")
st.markdown("Test if your Gemini API key is working")

st.markdown("---")

api_key = st.text_input(
    "Enter your Gemini API Key:", 
    type="password",
    placeholder="AIzaSy..."
)

if st.button("🧪 Test API Key", type="primary"):
    if not api_key:
        st.error("❌ Please enter an API key")
    else:
        st.info(f"📏 Key length: {len(api_key)} characters")
        st.info(f"🔤 Starts with: {api_key[:6]}...")
        st.info(f"🔤 Ends with: ...{api_key[-6:]}")
        
        if len(api_key) != 39:
            st.warning(f"⚠️ API key should be 39 characters, yours is {len(api_key)}")
        
        if not api_key.startswith("AIza"):
            st.warning("⚠️ API key should start with 'AIza'")
        
        st.markdown("---")
        st.markdown("### Testing API Key...")
        
        # Test Step 1
        step1 = st.empty()
        step1.info("⏳ Step 1/3: Configuring API...")
        
        try:
            genai.configure(api_key=api_key)
            step1.success("✅ Step 1/3: API configured successfully")
            
            # Test Step 2
            step2 = st.empty()
            step2.info("⏳ Step 2/3: Creating model...")
            
            try:
                model = genai.GenerativeModel('gemini-pro')
                step2.success("✅ Step 2/3: Model created successfully")
                
                # Test Step 3
                step3 = st.empty()
                step3.info("⏳ Step 3/3: Testing generation...")
                
                try:
                    response = model.generate_content("Say 'Hello! API is working!'")
                    
                    if response and response.text:
                        step3.success("✅ Step 3/3: Response received successfully")
                        
                        st.markdown("---")
                        st.success("### 🎉 SUCCESS! Your API key is working!")
                        
                        st.markdown("**Response from Gemini:**")
                        st.info(response.text)
                        
                        st.markdown("---")
                        st.markdown("### ✅ What to do next:")
                        st.write("1. Copy this API key")
                        st.write("2. Go to your DocuMind AI app")
                        st.write("3. Paste it in the sidebar")
                        st.write("4. Click Configure")
                        
                        st.balloons()
                    else:
                        step3.error("❌ Step 3/3: Empty response from API")
                        st.error("The API responded but with no text")
                        
                except Exception as e3:
                    step3.error(f"❌ Step 3/3: Failed - {str(e3)}")
                    st.error("**Error Details:**")
                    st.code(str(e3))
                    
                    st.markdown("### 🔍 Possible Issues:")
                    error_msg = str(e3).lower()
                    
                    if "quota" in error_msg or "429" in error_msg:
                        st.warning("**Quota Exceeded**")
                        st.write("- You've hit the free tier limit")
                        st.write("- Wait 5-10 minutes and try again")
                        st.write("- Or enable billing in Google Cloud Console")
                    elif "permission" in error_msg or "403" in error_msg:
                        st.warning("**Permission Denied**")
                        st.write("1. Go to: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com")
                        st.write("2. Click ENABLE")
                        st.write("3. Wait 2 minutes")
                        st.write("4. Try again")
                    else:
                        st.warning("**Unknown Error**")
                        st.write("Check the error message above")
                        
            except Exception as e2:
                step2.error(f"❌ Step 2/3: Failed - {str(e2)}")
                st.error("**Error Details:**")
                st.code(str(e2))
                st.warning("Model creation failed. This usually means API configuration issue.")
                
        except Exception as e1:
            step1.error(f"❌ Step 1/3: Failed - {str(e1)}")
            st.error("**Error Details:**")
            st.code(str(e1))
            
            st.markdown("### 🔍 Possible Issues:")
            error_msg = str(e1).lower()
            
            if "invalid" in error_msg or "api_key" in error_msg:
                st.warning("**Invalid API Key**")
                st.write("1. Go to: https://aistudio.google.com/app/apikey")
                st.write("2. Create a NEW API key")
                st.write("3. Copy the ENTIRE key (39 characters)")
                st.write("4. Paste it here and test again")
            else:
                st.warning("**Configuration Error**")
                st.write("Check the error message above")

st.markdown("---")
st.markdown("### 📚 How to get API key:")
st.write("1. Visit: https://aistudio.google.com/app/apikey")
st.write("2. Sign in with Google")
st.write("3. Click 'Create API key'")
st.write("4. Copy the key (starts with AIza)")
st.write("5. Paste it above and click Test")
