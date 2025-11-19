import streamlit as st
import pandas as pd
import tempfile
import uuid
import os
import re

from helper_dev import (
    read_documents,
    create_jd_summary_prompt,
    create_resume_extraction_prompt,
    get_ratings_prompt,
    generate_response,
    generate_response_4o,
    convert_to_json,
    get_connection,
    exec_query,
    get_embedding,
    get_standard_country_name,
    get_tables
)

rating_pattern = re.compile(r"Final Rating\s*:\s*(\d+(\.\d+)?)")

# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------

st.set_page_config(page_title="Resume Scoring + DB Saving", layout="wide")
st.title("Resume Scoring System (with Database Saving + Detailed View)")

st.markdown("""
This application:
- Extracts resume details  
- Summarizes JD  
- Scores resumes using your EXACT helper_dev scoring logic  
- Saves candidate & embedding into PostgreSQL  
- Shows extraction + scoring output like before  
""")

# -----------------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------------

with st.sidebar:
    st.header("Candidate Additional Inputs")
    country_input = st.text_input("Country", "")
    hourlyrate_input = st.text_input("Hourly Rate", "")
    notice_input = st.text_input("Notice Period", "")
    visa_input = st.text_input("Visa Status", "")

    vendor_id = st.text_input("Vendor ID", "test_entire_summary123")

# -----------------------------------------------------------
# JOB DESCRIPTION INPUT
# -----------------------------------------------------------

st.subheader("Step 1: Enter Job Description")

jd_text = st.text_area("Paste Job Description here:", height=200)

jd_file = st.file_uploader(
    "Or upload JD file (txt/pdf/doc/docx):",
    type=['txt', 'pdf', 'doc', 'docx'],
    accept_multiple_files=False
)

if jd_file is not None and not jd_text.strip():
    tmpd = tempfile.mkdtemp()
    jd_path = os.path.join(tmpd, jd_file.name)
    with open(jd_path, "wb") as f:
        f.write(jd_file.getbuffer())
    jd_text = read_documents(jd_path)
    st.success("JD file loaded successfully.")

# -----------------------------------------------------------
# RESUME UPLOAD
# -----------------------------------------------------------

st.subheader("Step 2: Upload Resume Files")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF, DOCX, DOC):",
    type=['pdf', 'docx', 'doc'],
    accept_multiple_files=True
)

# -----------------------------------------------------------
# PROCESSING + SCORING + DB SAVING
# -----------------------------------------------------------

if st.button("Process & Save to DB"):
    if not jd_text.strip():
        st.error("Please provide a Job Description first.")
    elif not uploaded_files:
        st.error("Please upload one or more resume files.")
    else:
        st.info("Processing... Please wait.")

        # DB Connection
        cursor, engine, connection = get_connection()
        CANDIDATE_TABLE, VECTOR_TABLE, JD_TABLE = get_tables(vendor_id)

        # ------------------------------
        # 1. JD SUMMARY
        # ------------------------------
        jd_prompt = create_jd_summary_prompt(jd_text)
        jd_response = generate_response(jd_prompt)
        jd_json = convert_to_json(jd_response)
        jd_summary = jd_json.get("summary", jd_text)

        st.subheader("JD Summary")
        st.write(jd_summary)

        results = []

        # ------------------------------
        # PROCESS EACH RESUME
        # ------------------------------
        for uploaded in uploaded_files:
            st.markdown(f"### Processing: **{uploaded.name}**")

            tmpd = tempfile.mkdtemp()
            resume_path = os.path.join(tmpd, uploaded.name)
            with open(resume_path, "wb") as f:
                f.write(uploaded.getbuffer())

            resume_text = read_documents(resume_path)

            # --------------------------
            # Extract Resume JSON
            # --------------------------
            extraction_prompt = create_resume_extraction_prompt(resume_text)
            extraction_response = generate_response(extraction_prompt)
            resume_json = convert_to_json(extraction_response)

            # Standardize location
            location_resume = get_standard_country_name(resume_json.get("location", ""))

            # --------------------------
            # Add user inputs
            # --------------------------
            extra_info = f"""
            Country: {country_input}
            HourlyRate: {hourlyrate_input}
            NoticePeriod: {notice_input}
            VisaStatus: {visa_input}
            """

            resume_with_fields = resume_text + "\n\n" + extra_info

            # --------------------------
            # Score Resume
            # --------------------------
            scoring_prompt = get_ratings_prompt(resume_with_fields, jd_summary)
            scoring_response = generate_response_4o(scoring_prompt)

            match = rating_pattern.search(scoring_response)
            final_score = match.group(1) if match else "Not Found"

            # --------------------------
            # Embedding
            # --------------------------
            embedding = get_embedding(resume_text)

            # --------------------------
            # SAVE TO DB (exact as backend)
            # --------------------------

            # Create candidate table (if needed)
            create_candidate_table_query = f"""
            CREATE TABLE IF NOT EXISTS {CANDIDATE_TABLE} (
                candidate_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                email VARCHAR,
                location VARCHAR,
                designation VARCHAR,
                summary TEXT,
                resume TEXT,
                hourlyrate VARCHAR,
                visastatus VARCHAR,
                noticeperiod VARCHAR,
                country VARCHAR
            );
            """
            exec_query(cursor, create_candidate_table_query)

            candidate_id = str(uuid.uuid4())

            insert_candidate_query = f"""
            INSERT INTO {CANDIDATE_TABLE}
            (candidate_id, name, email, location, designation, summary,
             resume, hourlyrate, visastatus, noticeperiod, country)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """

            insert_candidate_values = (
                candidate_id,
                resume_json.get("name", "Not Mentioned"),
                resume_json.get("email", "Not Mentioned"),
                location_resume,
                resume_json.get("designation", "Not Mentioned"),
                resume_json.get("summary", "Not Mentioned"),
                resume_text,
                hourlyrate_input,
                visa_input,
                notice_input,
                country_input
            )

            exec_query(cursor, insert_candidate_query, insert_candidate_values)

            # Vector table
            create_vector_table_query = f"""
            CREATE TABLE IF NOT EXISTS {VECTOR_TABLE}
            (id VARCHAR PRIMARY KEY, embedding vector(1000));
            """
            exec_query(cursor, create_vector_table_query)

            insert_vector_query = f"""
            INSERT INTO {VECTOR_TABLE} (id, embedding) VALUES (%s, %s);
            """
            exec_query(cursor, insert_vector_query, (candidate_id, embedding))

            connection.commit()

            # Save for UI display
            results.append({
                "filename": uploaded.name,
                "candidate_id": candidate_id,
                "name": resume_json.get("name", "Not Mentioned"),
                "email": resume_json.get("email", "Not Mentioned"),
                "final_score": final_score,
                "resume_json": resume_json,
                "scoring_output": scoring_response
            })

        # ------------------------------
        # DISPLAY SUMMARY TABLE
        # ------------------------------
        st.subheader("Saved Candidates")
        df = pd.DataFrame([
            {
                "File": r["filename"],
                "Candidate ID": r["candidate_id"],
                "Name": r["name"],
                "Email": r["email"],
                "Score": r["final_score"]
            }
            for r in results
        ])
        st.dataframe(df)

        # ------------------------------
        # EXPANDERS FOR FULL DETAILS
        # ------------------------------
        for r in results:
            with st.expander(f"{r['filename']} → Score: {r['final_score']}"):
                st.write("### Extracted Resume JSON")
                st.json(r["resume_json"])

                st.write("### Scoring Output")
                st.markdown(r["scoring_output"])

                st.write("### Resume Text (first 2000 chars)")
                st.write(r["resume_json"].get("summary", "No summary found."))

        st.success("All resumes processed and saved successfully!")
