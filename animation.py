# Description: This file contains the code to create an animated green light using HTML and CSS.
def golive():
    import streamlit as st

    # Define the HTML and CSS for the animation
    css = """
    <style>
    @keyframes flicker {
        0% { opacity: 0.2; }
        50% { opacity: 1; }
        100% { opacity: 0.2; }
    }

    .container {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .green-light {
        width: 10px;
        height: 10px;
        background-color: green;
        border-radius: 50%;
        animation: flicker 2s infinite;
    }
    </style>
    """

    # Render the CSS using st.markdown()
    st.markdown(css, unsafe_allow_html=True)

    # Render the animated green light inside a container
    return st.markdown('<div class="container"><div class="green-light"></div></div>', unsafe_allow_html=True)
