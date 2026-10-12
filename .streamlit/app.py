import streamlit as st

st.set_page_config(page_title="ORYN Diagnostic", page_icon="🔍")

st.title("🔍 ORYN System Diagnostic")
st.write("Testing your modular engine...")

# Check 1: Sandbox Module
try:
    import sandbox
    st.success("✅ sandbox.py is working perfectly!")
except Exception as e:
    st.error(f"❌ sandbox.py error: {e}")

# Check 2: Tools Module
try:
    import tools
    st.success("✅ tools.py is working perfectly!")
except Exception as e:
    st.error(f"❌ tools.py error: {e}")

# Check 3: Agent Module
try:
    import agent
    st.success("✅ agent.py is working perfectly!")
except Exception as e:
    st.error(f"❌ agent.py error: {e}")
