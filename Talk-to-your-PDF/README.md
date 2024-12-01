
# Talk to your PDF


## Overview

"Talk to your PDF" is an interactive app that allows users to upload PDF documents and ask questions about their content. Utilizing advanced NLP techniques like retrieval augmented generation (RAG), the app extracts text from the PDFs, generates embeddings of the PDF text and question, and provides relevant responses to user queries without the need for manual document navigation. This app is built from scratch, no framework (like LangChain) used.


## Features

- PDF text extraction and embedding generation for efficient information retrieval.
- User query processing with intent detection to ensure relevant and safe interactions.
- Database storage of PDF embeddings for quick access and response generation.
- Integration with Google Drive for secure and convenient file management.
- Clean and intuitive interface

## Tech Stack

- Python 3.11+
- Streamlit for the web interface
- OpenAI for embeddings and intent detection (using the moderation model) and response generation (using GPT-4 turbo)
- PostgreSQL with Supabase for database management
- pgvector PostgreSQL extension to handle vectors embedding
- Google Drive API for the PDF storage
- Requests and Tempfile for handling HTTP requests and temporary file management
- SQLAlchemy for connecting to the PostgreSQL database using Python to run SQL queries


## How the app Works



#### Step 1: File Upload

The user begins by uploading a PDF document through the application's interface. The backend receives this file and prepares it for text processing.

#### Step 2: Pre-run Service

The `PreRunProcessor` class takes charge by extracting text from the PDF. It then generates embeddings for the extracted text using the OpenAI API. These embeddings are numerical representations that encapsulate the meaning of the text segments, making them suitable for comparison and retrieval tasks.

#### Step 3: Intent Service

When the user submits a query about the PDF's content, the `IntentService` class performs two critical functions:

- Malicious Intent Detection: Utilizes OpenAI's moderation model to ensure the query does not contain inappropriate content.
- Relatedness Check: Converts the user's query into embeddings and checks if the query is relevant to the content of the PDF. This is done by comparing the query embeddings against the stored text embeddings from the PDF.


#### Step 4: Information Retrieval Service

Upon establishing the relevance of the query to the PDF content, the `InformationRetrievalService` class employs a cosine similarity search to find the most pertinent text segment that matches the query embeddings. Cosine similarity is a metric used to measure how similar the embeddings are, thus identifying the segment of the PDF that best addresses the user's query.

#### Step 5: Response Service

Finally, the `ResponseService` class uses the information retrieved to generate a response to the user's query. It leverages OpenAI's ChatCompletion API to formulate an answer that is not only relevant but also articulated in a clear and informative manner.

## Quick Start

This guide will walk you through setting up the "Talk to Your PDF" project from start to finish. Follow these steps to clone the repository, set up your environment, and start interacting with your PDFs in a new way.

0. **Pre-Requisites**

    Before starting, ensure you have the following:
      - **Openai API account**: Needed for the OpenAI API key.
      - **Python**: Make sure Python is installed on your machine.
      - **Google Cloud Platform (GCP) Account**: Needed for Google Drive API setup.
      - **Supabase Account**: Required for database setup.


1. **Create a Virtual Environment**

    Isolate your project's Python dependencies by creating a virtual environment

    ```bash
    python -m venv venv
    source venv/bin/activate # Unix/macOS
    venv\Scripts\activate # Windows
    ```

2. **Install dependencies**
   
   Install the necessary Python packages for the project

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables Setup**
   
   Add to your `venv` file in the project directory to securely store your environment variables. Initially, it will look like this

   ```bash
    GOOGLE_APPLICATION_CREDENTIALS=path/to/your/google-credentials.json
    SUPABASE_POSTGRES_URL=your_supabase_postgres_url_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```

4. **Google drive API setup**
    
   - **Create a Project in Google Cloud Platform (GCP):**
     - Go to the [Google Cloud Console](https://console.cloud.google.com/).
     - Click on "Select a project" at the top of the dashboard and then "New Project".
     - Enter a project name and click "Create".

   - **Enable Google Drive API:**
     - In the dashboard of your new project, navigate to "APIs & Services" > "Dashboard".
     - Click on "+ ENABLE APIS AND SERVICES".
     - Search for "Google Drive API" and enable it for your project.

   - **Create Credentials:**
     - Go to "APIs & Services" > "Credentials".
     - Click on "Create credentials" and select "Service account".
     - Fill out the service account details and grant it a role of "Editor" (or a more restricted role if you know the specific permissions you need).
     - Click "Create" and then "Done".

   - **Generate a Key for the Service Account:**
     - Find your new service account in the "Credentials" page and click on it.
     - Go to the "Keys" tab.
     - Click on "Add Key" and choose "Create new key".
     - Select "JSON" as the key type and click "Create".
     - The JSON key file will be automatically downloaded. This file contains your API key.
     - Important: Update the GOOGLE_APPLICATION_CREDENTIALS in your .env file with the path to this downloaded JSON file.

   - **Configure the Python Script:**
     - Open the JSON key file with a text editor and find the `client_email` and `private_key` fields.
     - In your Python script or environment, set the `client_email` and `private_key` as environment variables, or directly insert them into the `credentials_info` dictionary within the `upload_to_google_drive` function.

   - **Set Environment Variables:**
     - For security reasons, it's best practice to use environment variables for sensitive information like API keys.
     - You can set environment variables in your system, or use a `.env` file and load them with `load_dotenv()` in your script.
     - Set `GOOGLE_APPLICATION_CREDENTIALS` as an environment variable pointing to the path of the downloaded JSON key file.

   - **Install Required Libraries:**
     - Make sure that all required Python libraries, including `oauth2client`, `PyDrive`, and `google-auth`, are installed in your environment. You can usually install these using `pip install`.

   - **Use the Credentials in the Script:**
     - In the script, ensure that the `upload_to_google_drive` function is correctly configured to authenticate with Google using the service account credentials.

   - **Test the Setup:**
     - Run your script in a secure, local development environment first to ensure the Google Drive API is being called correctly and the file is being uploaded.
     - Make sure to handle the `credentials_dict` properly without exposing your private keys in the source code if you're going to share the script or use it in a public setting.

5. **Supabase database setup**

   - **Sign Up or Log In to Supabase**
      - Navigate to [Supabase](https://supabase.io) and sign in or create a new account.
      - Once logged in, create a new project by clicking on the 'New Project' button.

   - **Project Configuration**
      - Fill in the necessary details for your project, such as the name and password.
      - Wait for the project to be fully deployed. This might take a couple of minutes.

   - **Database Setup**
      - Once the project is ready, go to the SQL editor in the Supabase dashboard of your project.
      - You can run SQL commands here to set up your database schemas and extensions.

   - **Install pgvector Extension**
      - Execute the following SQL command to add the `pgvector` extension to your database:
        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        ```
      - This command sets up the database to handle vector data types which are required for the embeddings.

   - **Create Table with Vector Column**
      - Create a table that will store your embeddings with a vector column by running:
        ```sql
        CREATE TABLE pdf_holder (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding VECTOR(3072)
        );
        ```
      - This table will be used to store the text from the PDFs and their corresponding embeddings.

   - **Retrieve Supabase Credentials**
      - After setting up your table, go to the 'Settings' > 'Database' section of your project's dashboard.
      - Here you can find the `SUPABASE_POSTGRES_URL` which is required for connecting to the database from your application.

   - **Set Up Environment Variables**
      - In your application environment, set the `SUPABASE_POSTGRES_URL` as an environment variable.
      - This is essential for your application to connect and interact with your Supabase database securely.

   - **Integrate with the Application**
      - Use the `create_engine` function from the `sqlalchemy` library to connect to your Supabase database using the URL from the previous step.
      - Utilize the database session to execute your operations, such as storing and retrieving vector data.

   - **Test the Database Connection**
      - Before proceeding, ensure that your application can connect to the Supabase database and perform operations.
      - Try inserting a dummy record or retrieving data from the `pdf_holder` table as a test.

   - **Deploy Your Application**
       - After confirming the database connection, you can proceed to deploy your application.
       - Monitor the application logs to ensure there are no connection issues with the Supabase database.
6. **OpenAI API Setup**

    - Obtain your OpenAI API key from the OpenAI API platform.
    - Fill in the `OPENAI_API_KEY` in your `.env` file with your key.

7. **Run the app locally**

    Go to the directory that contains the app and run

   ```bash
   streamlit run streamlit_app.py
   ```




