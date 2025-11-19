import os
import pandas as pd
import json
import shutil
import warnings
import re
import logging
import datetime

import uuid
from typing import List
from typing import Optional
# from pyngrok import ngrok
import uvicorn

from helper_dev import create_resume_extraction_prompt, create_jd_summary_prompt, create_jd_creation_prompt, generate_response, convert_to_json, \
                   get_embedding, read_documents, get_connection, exec_query, get_ratings_prompt, generate_response_4o, get_tables, create_linkedin_validator_prompt, get_standard_country_name, SCORE_LIMIT

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio # Naga commented this
from collections import defaultdict
import time

# Set up logging configuration at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Global dictionary to store processing status
processing_status = defaultdict(lambda: {
    "total": 0,
    "processed": 0,
    "completed": False,
    "failed_files": []
})

def process_files_background(vendor_id: str, location: str, hourlyrate: str,
                                 visastatus: str, noticeperiod: str, files: List[dict],
                                 task_id: str, temp_dir: str):
    logger.info(f"Starting background processing for task {task_id}")
    logger.info(f"Number of files to process: {len(files)}")

    try:
        # Update total files to process
        processing_status[task_id]["total"] = len(files)
        result_df = pd.DataFrame()

        # Get database connection and tables
        cursor, engine, connection = get_connection()
        CANDIDATE_TABLE, VECTOR_TABLE, JD_TABLE = get_tables(vendor_id=vendor_id)
        logger.info(f"Database connection established for task {task_id}")

        def process_file(file_data):
            try:
                start_time = time.time()
                filename = file_data['filename']
                filepath = file_data['filepath']

                logger.info(f"Processing file {filename} for task {task_id}")

                # Your existing file processing logic here
                text = read_documents(filepath)

                # Generate response for resume
                prompt_template = create_resume_extraction_prompt(text)
                response_text = generate_response(prompt_template)  # Now returns string directly

                # Convert response to JSON
                json_data = convert_to_json(response_text)

                # Create DataFrame
                candidate_df = pd.DataFrame([json_data])
                state = candidate_df['state']
                candidate_df = candidate_df.drop(columns=['state'])
                location_resume = get_standard_country_name(candidate_df['location'].values[0])
                country_name = get_standard_country_name(location_resume)
                print(f'Resume Location : {location_resume}')
                # Add additional fields
                uu_id = str(uuid.uuid4())
                candidate_df['candidate_id'] = uu_id
                candidate_df['location'] = location_resume
                candidate_df['hourlyrate'] = hourlyrate
                candidate_df['visastatus'] = visastatus
                candidate_df['noticeperiod'] = noticeperiod
                candidate_df['country'] = country_name

                # Get vector embedding
                vector_resume = get_embedding(text)

                # Create and insert into tables
                create_table_query = f"""
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

                insert_query = f"INSERT INTO {CANDIDATE_TABLE} (candidate_id, name, email, location, designation, summary, resume, hourlyrate, visastatus, noticeperiod, country) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
                values = (uu_id, json_data['name'], json_data['email'], location_resume, json_data['designation'], json_data['summary'], text, hourlyrate, visastatus, noticeperiod, country_name)

                exec_query(cursor, create_table_query)
                exec_query(cursor, insert_query, values)

                create_vector_query = f"CREATE TABLE IF NOT EXISTS {VECTOR_TABLE} (id VARCHAR PRIMARY KEY, embedding vector(1000));"
                insert_vector_query = f"INSERT INTO {VECTOR_TABLE} (id, embedding) VALUES (%s, %s);"
                exec_query(cursor, create_vector_query)
                exec_query(cursor, insert_vector_query, (uu_id, vector_resume,))
                connection.commit()

                end_time = time.time()
                logger.info(f"File {filename} processed in {end_time - start_time:.2f} seconds for task {task_id}")

                # Update processed count and return the DataFrame with file information
                processing_status[task_id]["processed"] += 1
                candidate_df['original_filename'] = filename
                candidate_df['saved_filename'] = file_data['timestamped_filename']
                candidate_df['file_path'] = file_data['permanent_path']
                candidate_df['state'] = state
                return candidate_df

            except Exception as e:
                logger.error(f"Error processing file {filename} for task {task_id}: {str(e)}")
                processing_status[task_id]["failed_files"].append({
                    "filename": filename,
                    "error": str(e)
                })
                return None

        # Process files using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            logger.info(f"Starting parallel file processing for task {task_id}")
            results = list(executor.map(process_file, files))
            logger.info(f"Completed parallel file processing for task {task_id}")

        # Combine results
        result_df = pd.concat([df for df in results if df is not None], ignore_index=True)

        # Update status
        processing_status[task_id]["completed"] = True
        result_records = result_df.to_dict('records') if not result_df.empty else None

        # Add file processing summary
        processing_status[task_id].update({
            "result": result_records,
            "processed_files": [
                {
                    "original_filename": file['filename'],
                    "saved_filename": file['timestamped_filename'],
                    "file_path": file['permanent_path']
                } for file in files
            ]
        })

        logger.info(f"Background processing completed for task {task_id}")
        logger.info(f"Processed {processing_status[task_id]['processed']} files, {len(processing_status[task_id]['failed_files'])} failed")

        # Only clean up temporary processing directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory for task {task_id}")

    except Exception as e:
        logger.error(f"Error in background processing for task {task_id}: {str(e)}")
        processing_status[task_id]["completed"] = True
        processing_status[task_id]["error"] = str(e)

        # Only clean up temporary processing directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory for task {task_id} after error")

@app.post("/upload_resume/")
async def upload_files_and_extract_text(
    background_tasks: BackgroundTasks,
    vendor_id: str,
    location: str,
    hourlyrate: str,
    visastatus: str,
    noticeperiod: str,
    files: List[UploadFile] = File(...)
):
    task_id = str(uuid.uuid4())
    start_time = time.time()

    # Create a temporary directory for file processing
    temp_dir = f"temp_{task_id}"
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize processing status
    processing_status[task_id] = {
        "total": 0,
        "processed": 0,
        "failed_files": [],
        "completed": False
    }

    # Save files to temporary directory
    file_paths = []
    save_errors = []

    async def save_file(file):
        try:
            file_start_time = time.time()
            # Create a timestamp for the file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create a filename with timestamp
            base_name, ext = os.path.splitext(file.filename)
            timestamped_filename = f"{base_name}_{timestamp}{ext}"

            # Create the permanent storage directory if it doesn't exist
            storage_dir = "/var/www/html/public/user-uploads/documents"
            try:
                os.makedirs(storage_dir, exist_ok=True)
                logger.info(f"Storage directory ensured at {storage_dir}")
            except PermissionError as e:
                logger.error(f"Permission error creating directory {storage_dir}: {str(e)}")
                raise Exception(f"Permission denied to create/access storage directory. Please ensure proper permissions are set for {storage_dir}")

            # Save to both temporary and permanent locations
            temp_file_path = os.path.join(temp_dir, timestamped_filename)
            permanent_file_path = os.path.join(storage_dir, timestamped_filename)

            # Save the file content
            contents = await file.read()
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(contents)
                # Also save to permanent location
                with open(permanent_file_path, "wb") as f:
                    f.write(contents)
            except PermissionError as e:
                logger.error(f"Permission error saving file {permanent_file_path}: {str(e)}")
                raise Exception(f"Permission denied to write file. Please ensure proper permissions are set for {storage_dir}")

            file_paths.append({
                'filename': file.filename,
                'filepath': temp_file_path,
                'permanent_path': permanent_file_path,
                'timestamped_filename': timestamped_filename
            })

            file_end_time = time.time()
            print(f"File {file.filename} saved in {file_end_time - file_start_time:.2f} seconds")

        except Exception as e:
            print(f"Error saving file {file.filename}: {str(e)}")
            save_errors.append({
                "filename": file.filename,
                "error": str(e)
            })

    try:
        # Create tasks for all files
        tasks = [save_file(file) for file in files]

        # Wait for all files to be saved
        print(f"Starting parallel file save at {time.time()}")
        await asyncio.gather(*tasks)
        print(f"Finished parallel file save at {time.time()}")

        # Check if there were any errors
        if save_errors:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return {
                "status": "error",
                "message": "Some files failed to save",
                "errors": save_errors
            }

        print(f"Starting background process at {time.time()}")

        # Start the background task directly
        background_tasks.add_task(
            process_files_background,
            vendor_id,
            location,
            hourlyrate,
            visastatus,
            noticeperiod,
            file_paths,
            task_id,
            temp_dir
        )

        end_time = time.time()
        print(f"Returning response at {time.time()}")
        print(f"Total upload time: {end_time - start_time:.2f} seconds")
        return {"task_id": task_id, "message": "Processing started"}

    except Exception as e:
        # Only clean up temporary processing directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return {
            "status": "error",
            "message": f"Error in upload process: {str(e)}"
        }

@app.get("/process_status/{task_id}")
async def get_process_status(task_id: str):
    async def status_generator():
        previous_status = None

        while not processing_status[task_id]["completed"]:
            current_status = {
                "total": processing_status[task_id]["total"],
                "processed": processing_status[task_id]["processed"],
                "failed_files": processing_status[task_id]["failed_files"],
                "completed": processing_status[task_id]["completed"]
            }

            # Only yield if status has changed
            if previous_status != current_status:
                yield f"data: {json.dumps(current_status)}\n\n"
                previous_status = current_status.copy()

            await asyncio.sleep(1)  # Check every 500ms

        # Send final status if it's different from the last sent status
        final_status = {
            "total": processing_status[task_id]["total"],
            "processed": processing_status[task_id]["processed"],
            "failed_files": processing_status[task_id]["failed_files"],
            "completed": True,
            "result": processing_status[task_id].get("result", None),
            "error": processing_status[task_id].get("error", None)
        }

        if previous_status != final_status:
            yield f"data: {json.dumps(final_status)}\n\n"

        # Cleanup status after completion
        if processing_status[task_id]["completed"]:
            del processing_status[task_id]

    return StreamingResponse(
        status_generator(),
        media_type="text/event-stream"
    )

@app.get('/index')
async def home():
    return "Hello World"

@app.post("/linkedin_vaidator/")
async def upload_files_and_validate(
    candidate_id: str,
    vendor_id: str,
    file: UploadFile = File(...)
    ):
    linkedin_file_location = os.path.join(".", file.filename)
    with open(linkedin_file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cursor, engine, connection = get_connection()
    CANDIDATE_TABLE, VECTOR_TABLE, JD_TABLE = get_tables(vendor_id=vendor_id)

    get_candidate_resume_query = f"SELECT resume from {CANDIDATE_TABLE} WHERE candidate_id = '{candidate_id}'"
    df_candidate = pd.read_sql_query(get_candidate_resume_query, connection)
    resume_text = df_candidate['resume'].values
    linkedin_text = read_documents(linkedin_file_location)

    prompt_template = create_linkedin_validator_prompt(resume_text, linkedin_text)
    response = generate_response(prompt_template)
    print(f"Response: {response}")

    # print(response.choices[0].message.content)

    comparision = convert_to_json(response)['Compare']

    return {
        "status": "success",
        "code": 200,
        "data" : {comparision}
    }


@app.post("/upload_jd/")
async def upload_jd_def(job_description: str, vendor_id: str):
    try:
        # Get database connection
        cursor, engine, connection = get_connection()
        CANDIDATE_TABLE, VECTOR_TABLE, JD_TABLE = get_tables(vendor_id=vendor_id)

        # Create JD summary
        prompt_template = create_jd_summary_prompt(job_description)
        response_text = generate_response(prompt_template)  # Now returns string directly
        print(f"JD : Response {response_text}")
        json_data = convert_to_json(response_text)
        summary = json_data.get('summary', response_text)  # Fallback to full response if no summary field

        # Create JD table if not exists
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {JD_TABLE} (
            jd_id VARCHAR PRIMARY KEY,
            job_description TEXT,
            summary TEXT
        );
        """
        exec_query(cursor, create_table_query)

        # Insert JD
        jd_id = "jd-" + str(uuid.uuid4())
        insert_query = f"INSERT INTO {JD_TABLE} (jd_id, job_description, summary) VALUES (%s, %s, %s);"
        values = (jd_id, job_description, summary)
        exec_query(cursor, insert_query, values)
        connection.commit()

        return {
            "status": "success",
            "code": 200,
            "data": {
                "jd_id": jd_id,
                "summary": summary
            }
        }
    except Exception as e:
        logger.error(f"Error uploading JD: {str(e)}")
        return {
            "status": "error",
            "code": 505,
            "error": {
                "message": str(e)
            }
        }

@app.post("/match_profiles/")
async def match_profiles_def(jd_id: str, vendor_id: str, no_of_profiles: int, jd_location: str, country: Optional[str] = None):
    try:
        CANDIDATE_TABLE, VECTOR_TABLE, JD_TABLE = get_tables(vendor_id=vendor_id)
        cursor, engine, connection = get_connection()

        # Use async/await to execute database queries concurrently
        async def fetch_summary_and_embedding():
            query_get_embeddings = f"SELECT summary FROM {JD_TABLE} WHERE jd_id = '{jd_id}'"
            cursor.execute(query_get_embeddings)
            summary = cursor.fetchall()[0][0]
            return summary, get_embedding(summary)

        async def fetch_similar_vectors(embedding):
            query_similar_vectors = f"SELECT id, 1 - (embedding <=> '{embedding}') AS similarity_score FROM {VECTOR_TABLE} ORDER BY embedding <=> '{embedding}'"
            return pd.read_sql_query(query_similar_vectors, connection)

        summary, embedding = await fetch_summary_and_embedding()
        df_scores = await fetch_similar_vectors(embedding)

        match_ids = df_scores['id'].tolist()
        ids_string = "(" + ",".join(f"'{id}'" for id in match_ids) + ")"
        get_profiles_query = f"""SELECT candidate_id, name, email, location, designation, summary, resume, hourlyrate, visastatus, noticeperiod, country from {CANDIDATE_TABLE} WHERE candidate_id IN {ids_string}"""

        df_profiles = pd.read_sql_query(get_profiles_query, connection)
        print('Profiles')
        print(df_profiles)
        df = pd.merge(df_profiles, df_scores, left_on="candidate_id", right_on="id", how="inner")

        print('Locations')
        print(df_profiles['location'])

        # df = df.sort_values(by='similarity_score', ascending=False).head(no_of_profiles)

        # Merge columns for efficient processing
        df['all'] = 'Location :' + df['location'] + ' Hourlyrate:' + df['hourlyrate'] + ' Visastatus:' + df['visastatus'] + ' NoticePeriod:' + df['noticeperiod'] + ' ' + df['resume']
        location_jd = get_standard_country_name(jd_location)
        df = df[df['location'] == location_jd]
        print(f'Location_jd : {location_jd}')
        print('DataFrame')
        print(df.head())
        if df.empty:
            pass
        else:
            # Use thread pooling for parallel processing of LLM calls
            def process_resume(resume):
                ratings_prompt = get_ratings_prompt(resume, summary)
                response_text = generate_response_4o(ratings_prompt)  # This now returns string content
                match = rating_pattern.search(response_text)
                return response_text, match.group(1).strip() if match else ""

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_resume, df['resume'].values))

            candidate_rating, final_rating = zip(*results)
            df['candidate_rating'] = candidate_rating

            df['final_score'] = [f"{score}%" for score in final_rating]

            df = df.head(no_of_profiles)
            df = df.sort_values(by='similarity_score', ascending=False)
            df = df.drop(columns=['summary', 'resume', 'similarity_score', 'all'])

            df['final_score_numeric'] = df['final_score'].str.replace('%', '').astype(float)
            df = df[df['final_score_numeric'] > SCORE_LIMIT]
            df = df.drop(columns=['final_score_numeric'])

        return {
            "status": "success",
            "code": 200,
            "data": json.loads(df.to_json(orient='records'))
        }
    except Exception as e:
        logger.error(f"Error matching profiles: {str(e)}")
        return {
            "status": "error",
            "code": 505,
            "error": {
                "message": str(e)
            }
        }

rating_pattern = re.compile(r"Final Rating\s*:\s*(\d+(\.\d+)?)")
@app.post("/generate_jd/")
async def generate_jd(job_description: str):
    try:
        prompt_template = create_jd_creation_prompt(job_description)
        response_text = generate_response(prompt_template)  # Now returns string directly
        return {
            "status": "success",
            "code": 200,
            "data": response_text
        }
    except Exception as e:
        logger.error(f"Error generating JD: {str(e)}")
        return {
            "status": "error",
            "code": 505,
            "error": {
                "message": str(e)
            }
        }

if __name__ == "__main__": # Naga added this
    uvicorn.run(app, host="127.0.0.1", port=8000)
    #uvicorn.run(app, port=5000, host="0.0.0.0",
            #ssl_keyfile="/etc/ssl/godaddy/dev.primehire.ai-PrivateKey.pem",  # Path to the private key
            #ssl_certfile="/etc/ssl/godaddy/dev.primehire.ai-certificate.crt",  # Path to the certificate
            #ssl_ca_certs="/home/ubuntu/gd_bundle-g2-g1.crt"
            #)
