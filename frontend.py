import streamlit as st
import requests


st.title("PDF Upload and Query App")

backend_url = "http://127.0.0.1:8000"

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])


if st.button("Upload PDF"):
    if uploaded_file is not None:
        try:

            pdf_bytes = uploaded_file.read()


            files = {"file": (uploaded_file.name, pdf_bytes, "application/pdf")}


            response = requests.post(f"{backend_url}/upload", files=files)


            if response.status_code == 200:
                st.success("PDF uploaded successfully!")
                st.write(response.json())
            else:
                st.error(f"Error from server: {response.status_code}")
                st.write(response.text)
        except Exception as e:
            st.error("An error occurred while uploading the PDF.")
            st.write(str(e))
    else:
        st.warning("Please upload a PDF file.")


query = st.text_input("Enter your query")


if st.button("Submit Query"):
    if query:
        try:

            data = {"question": query}

            response = requests.post(f"{backend_url}/ask", json=data)

            if response.status_code == 200:
                st.success("Response from server:")
                data=response.json()
                data=data['answer']
                st.write(data)
            else:
                st.error(f"Error from server: {response.status_code}")
                st.write(response.text)
        except Exception as e:
            st.error("An error occurred while processing your query.")
            st.write(str(e))
    else:
        st.warning("Please enter a query.")
