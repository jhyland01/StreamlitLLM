import streamlit as st

# use full width
st.set_page_config(layout="wide")

def main():
    # include llama emoji
    st.title("Local LLM Demo 🦙") 
    st.write("For demonstration purposes: here you can chat with different LLM models and use different features like Retrieval Augmented Generation.")
    st.write("Enjoy!")
    # insert a space
    st.markdown("----")

    # use cases side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(r"$\texttt{Pair\ on\ proprietary\ code\ in\ safety}$")
        st.image("icons/code_blur.png", use_column_width=True)
    with col2:
        st.latex(r"\text{Query sensitive documents}")
        st.image("icons/sensitive_info.jpg", use_column_width=True)
    with col3:
        st.subheader("Use the latest LLMs for free")
        st.image("icons/gifrecording.gif", use_column_width=True)

if __name__ == "__main__":
    main()
