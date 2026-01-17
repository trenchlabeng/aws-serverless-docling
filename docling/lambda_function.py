import json
import os
import boto3
import traceback
import logging
import textwrap
from urllib.parse import urlparse
from docling.chunking import HybridChunker
from openai import OpenAI
import re
import psycopg2

# IMPORTANT: This assumes your docling_parser.py has the VlmDocParser class
from docling_parser import VlmDocParser

# --- AWS CLIENTS AND CONFIGURATION ---
s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')
secrets_client = boto3.client('secretsmanager')

# --- ENVIRONMENT VARIABLES ---
SQS_QUEUE_URL = os.environ.get('SQS_QUEUE_URL')
OUTPUT_PREFIX = os.environ.get('OUTPUT_PREFIX', 'processed/')
LITELLM_PROXY_URL = os.environ.get('LITELLM_PROXY_URL')
DB_SECRET_NAME = os.environ.get('DB_SECRET_NAME')

# --- SHARED CLIENTS (Initialized once per container) ---
client = OpenAI(
    base_url=f"{LITELLM_PROXY_URL}/v1",
    api_key="not-needed"
)
db_creds = None
conn = None

# --- LOGGING SETUP ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# --- HELPER FUNCTIONS ---

def parse_s3_url(s3_url):
    """Extract bucket and key from s3:// URL"""
    parsed = urlparse(s3_url)
    return parsed.netloc, parsed.path.lstrip('/')

def extract_json_from_string(text: str) -> str:
    """Uses regex to find and extract the first valid JSON object from a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return ""

# --- DATABASE FUNCTIONS ---

def get_db_credentials():
    """Retrieves database credentials from AWS Secrets Manager."""
    global db_creds
    if db_creds is None:
        response = secrets_client.get_secret_value(SecretId=DB_SECRET_NAME)
        db_creds = json.loads(response['SecretString'])
    return db_creds

def get_db_connection():
    """Establishes or returns an active database connection."""
    global conn
    if conn is None or conn.closed:
        creds = get_db_credentials()
        conn = psycopg2.connect(
            host=creds['host'],
            port=creds['port'],
            database=creds['dbname'],
            user=creds['username'],
            password=creds['password']
        )
        logger.info("Successfully connected to the database.")
    return conn

def update_indexing_status(file_id, status, db_conn):
    """Updates the file's indexing status in the database."""
    logger.info(f"Updating status for file '{file_id}' to '{status}'")
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                """
                UPDATE files 
                SET indexing_status = %s, updated_at = NOW()
                WHERE id = %s;
                """,
                (status, file_id)
            )
            db_conn.commit()
    except Exception as e:
        logger.error(f"Failed to update indexing status for file '{file_id}': {e}")
        db_conn.rollback()


# --- MAIN LAMBDA HANDLER ---
def lambda_handler(event: dict, context):
    logger.info(f"Received SQS event to process document: {json.dumps(event)}")
    
    # 1. PARSE INCOMING SQS MESSAGE
    try:
        message_body = json.loads(event['Records'][0]['body'])
        file_id = message_body['fileId']
        case_id = message_body['caseId']
        bucket_name = message_body['s3Bucket']
        object_key = message_body['s3Key']
        s3_url = f"s3://{bucket_name}/{object_key}"
    except (KeyError, IndexError) as e:
        logger.error(f"Malformed SQS message. Error: {e}")
        return {'statusCode': 200, 'body': 'Malformed SQS message'}

    # FIX: Initialize db_conn to None before the try block
    db_conn = None
    try:
        # FIX: Establish DB connection first and use the local variable db_conn
        db_conn = get_db_connection()
        # FIX: Corrected arguments and using a more accurate initial status
        update_indexing_status(file_id, 'CHUNKING STARTED', db_conn)
        
        logger.info(f"Starting processing for FileId: {file_id}, CaseId: {case_id}")

        # 2. DOWNLOAD FILE FROM S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_bytes = response['Body'].read()
        logger.info(f"Downloaded {len(file_bytes)} bytes from {s3_url}")

        # 3. VLM DOCUMENT RECONSTRUCTION
        parser = VlmDocParser(bytes_content=file_bytes, vlm_proxy_url=LITELLM_PROXY_URL)
        vlm_doc = parser.reconstruct_document()

        # 4. VLM-POWERED METADATA EXTRACTION
        logger.info("Starting metadata extraction from VLM-reconstructed text...")
        extraction_prompt = textwrap.dedent("""
            You are an expert legal AI assistant specializing in personal injury law. 
            Your task is to analyze a case document and extract a comprehensive set of metadata in a structured JSON format. 
            Analyze the text provided and populate all relevant fields from the following schema. 
            If a field is not applicable or the information is not present, use null.
                                            
            JSON Schema to populate:
             {  "document_type": "Categorize the document (e.g., 'Medical Record', 'Police Accident Report', 'Insurance Correspondence')",
                "document_title": "The document's title", 
                "document_date": "The date the document was created (YYYY-MM-DD)", 
                "summary": "A 2-3 sentence summary of the document", 
                "entities": [ { "name": "Entity Name", "role": "Entity Role (e.g., Patient, Provider, Insurance Adjuster)", 
                "contact_info": "Any contact details", "policy_or_claim_number": "Associated ID numbers" } ], 
                "primary_date": "The main date of service or event (YYYY-MM-DD)", 
                "all_dates_mentioned": [ { "date": "YYYY-MM-DD", "description": "Context of the date" } ], 
                "event_description": "A narrative description of the document's main event", 
                "financial_summary": { "invoice_number": null, "service_date_range": null, "billed_amount": 0.0, "paid_amount": 0.0, "adjustments": 0.0, "outstanding_balance": 0.0, "is_lien": false }, 
                "causation_statements": [ "List of quotes linking injury to the incident" ], 
                "liability_indicators": [ "List of quotes related to fault" ], 
                "damages_evidence": [ { "type": "Category of damage (e.g., Pain and Suffering)", "quote_or_finding": "The supporting text" } ], 
                                    "prognosis_and_permanency": [ "List of quotes about the patient's future outlook" ],
                "clinical_breakdown": {
                    "subjective": "Patient complaints and history",
                    "objective": "Physical exam findings, test results, and observations",
                    "assessment": "Diagnoses and medical conclusions",
                    "plan": "Treatments, prescriptions, and follow-ups"
                }
            }

            Return only the JSON object.
        """) # Your detailed prompt here
        
        # FIX: Use the alias from your LiteLLM config, not the full model ID
        response = client.chat.completions.create(
            model="us.amazon.nova-2-lite-v1:0",
            messages=[{"role": "user", "content": f"{extraction_prompt}\n\nDOCUMENT_TEXT:\n{vlm_doc.export_to_markdown()}"}],
            temperature=0.0,
            max_tokens=8192
        )
        response_text = response.choices[0].message.content
        json_string = extract_json_from_string(response_text)
        if not json_string:
            raise ValueError(f"Could not find any JSON in the VLM response. Raw response was: {response_text}")
        
        extracted_metadata = json.loads(json_string)
        logger.info(f"Extracted metadata: {json.dumps(extracted_metadata)}")

        # 5. HYBRID CHUNKING
        logger.info("Starting document chunking...")
        # IMPROVEMENT: Set max_tokens to get your desired larger chunk size
        chunker = HybridChunker(merge_peers=True, max_tokens=1024)
        chunk_iter = chunker.chunk(dl_doc=vlm_doc)
        
        output_chunks = []
        for i, chunk in enumerate(chunk_iter):
            serialized_text = chunker.serialize(chunk)
            if not serialized_text.strip():
                continue

            # Extract page numbers from the chunk's provenance data
            page_numbers = sorted(list(set(
                prov.page_no
                for item in chunk.meta.doc_items
                for prov in item.prov if hasattr(prov, "page_no")
            )))

            # Extract bounding boxes for potential UI highlighting
            bboxes = [
                prov.bbox.as_tuple()
                for item in chunk.meta.doc_items
                for prov in item.prov if hasattr(prov, "bbox") and prov.bbox
            ]

            # Assemble the final chunk object
            output_chunks.append({
                "chunk_id": i,
                "text": serialized_text,
                "metadata": {
                    "page_numbers": page_numbers,
                    "bboxes": bboxes,
                    # This cleanly merges the document-level metadata into each chunk
                    **extracted_metadata
                }
            })
        logger.info(f"Generated {len(output_chunks)} chunks.")

        # 6. UPLOAD PROCESSED ARTIFACTS TO S3
        chunks_content = json.dumps(output_chunks, indent=2)
        
        base_name = os.path.basename(object_key)
        name_only = os.path.splitext(base_name)[0]
        target_folder = f"{OUTPUT_PREFIX}{case_id}/{name_only}"
        
        json_key = f"{target_folder}/{name_only}_chunks.json"
        # FIX: Added md_key for the markdown file
        meta_key = f"{target_folder}/metadata.json"
        
        s3_client.put_object(Bucket=bucket_name, Key=json_key, Body=chunks_content, ContentType='application/json')
        logger.info(f"Uploaded processed artifacts to S3 folder: {target_folder}")

        # 7. CREATE FINAL METADATA PAYLOAD & SEND SQS NOTIFICATION
        final_metadata_payload = {
            "caseId": case_id, "fileId": file_id, "original_document": s3_url,
            "processed_chunks": f"s3://{bucket_name}/{json_key}",
            "document_info": {"page_count": len(vlm_doc.pages), "chunk_count": len(output_chunks)},
            "status": "success"
        }
        s3_client.put_object(Bucket=bucket_name, Key=meta_key, Body=json.dumps(final_metadata_payload, indent=2))

        if SQS_QUEUE_URL:
            sqs_client.send_message(QueueUrl=SQS_QUEUE_URL, MessageBody=json.dumps(final_metadata_payload))
            logger.info(f"Sent completion message to SQS for fileId {file_id}")
        
        # FIX: Use the correct status and the local db_conn variable
        update_indexing_status(file_id, 'CHUNKING COMPLETE', db_conn)

        return {'statusCode': 200, 'body': json.dumps(final_metadata_payload)}

    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"FATAL ERROR processing fileId {file_id}: {str(e)}\n{stack_trace}")
        
        # FIX: Update status to FAILED on any error
        if db_conn:
            update_indexing_status(file_id, 'FAILED', db_conn)
            
        raise e
        
    finally:
        # FIX: CRITICAL - Always close the database connection
        global conn
        if conn and not conn.closed:
            conn.close()
            logger.info("Database connection closed.")