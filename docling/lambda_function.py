import json
import os
import boto3
import traceback
import logging
from urllib.parse import urlparse
from docling_parser import DoclingParser
from docling.chunking import HybridChunker

# Initialize Clients
s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')

# Configuration
SQS_QUEUE_URL = os.environ.get('SQS_QUEUE_URL')
# Output prefix can be empty if you want the bucket root to start with case_id
OUTPUT_PREFIX = os.environ.get('OUTPUT_PREFIX', 'processed/') 

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_s3_url(s3_url):
    """Extract bucket and key from s3:// URL"""
    parsed = urlparse(s3_url)
    return parsed.netloc, parsed.path.lstrip('/')

def lambda_handler(event: dict, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # 1. Parse Input
        if 'body' in event and isinstance(event['body'], str):
            body = json.loads(event['body'])
        else:
            body = event

        s3_url = body.get('s3Url')
        case_id = body.get('caseId') # Get Case ID
        
        # Validate Inputs
        if not s3_url:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Missing s3Url parameter'})}
        
        if not case_id:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Missing caseId parameter'})}

        bucket_name, object_key = parse_s3_url(s3_url)
        logger.info(f"Processing Case: {case_id} | Document: {object_key}")

        # 2. Download File from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_bytes = response['Body'].read()
        logger.info(f"Downloaded {len(file_bytes)} bytes")

        # 3. Process Document (Parsing)
        is_image_present = body.get('isImagePresent', False)
        
        parser = DoclingParser(
            bytes_content=file_bytes,
            is_image_present=is_image_present,
            is_md_response=False 
        )
        
        logger.info(f"Detected document type: {parser.doc_type}")
        
        conversion_result = parser.parse_documents()
        doc = conversion_result.document

        # 4. Hybrid Chunking
        chunker = HybridChunker(merge_peers=True)
        chunk_iter = chunker.chunk(dl_doc=doc)
        
        output_chunks = []
        for i, chunk in enumerate(chunk_iter):
            serialized_text = chunker.serialize(chunk)
            if not serialized_text.strip(): continue

            page_numbers = sorted(list(set(
                prov.page_no for item in chunk.meta.doc_items 
                for prov in item.prov if hasattr(prov, "page_no")
            )))
            
            bboxes = []
            for item in chunk.meta.doc_items:
                 for prov in item.prov:
                     if hasattr(prov, "bbox") and prov.bbox:
                         bboxes.append(prov.bbox.as_tuple())

            output_chunks.append({
                "chunk_id": i,
                "text": serialized_text,
                "metadata": {
                    "page_numbers": page_numbers,
                    "bboxes": bboxes,
                    "doc_items": [str(item.self_ref) for item in chunk.meta.doc_items]
                }
            })

        # 5. Prepare Output Content
        markdown_content = doc.export_to_markdown()
        chunks_content = json.dumps(output_chunks, indent=2)
        
        # 6. Upload Results to S3
        # Logic: processed/case_123/filename/filename.md
        base_name = os.path.basename(object_key)
        name_only = os.path.splitext(base_name)[0]
        
        # Construct the specific folder path for this case and document
        # Ensure OUTPUT_PREFIX ends with / if it exists, or handle empty string
        prefix = OUTPUT_PREFIX if OUTPUT_PREFIX.endswith('/') else f"{OUTPUT_PREFIX}/"
        target_folder = f"{prefix}{case_id}/{name_only}"
        
        md_key = f"{target_folder}/{name_only}.md"
        json_key = f"{target_folder}/{name_only}_chunks.json"
        meta_key = f"{target_folder}/metadata.json"
        
        # Upload Markdown
        s3_client.put_object(
            Bucket=bucket_name,
            Key=md_key,
            Body=markdown_content,
            ContentType='text/markdown'
        )
        
        # Upload Chunks JSON
        s3_client.put_object(
            Bucket=bucket_name,
            Key=json_key,
            Body=chunks_content,
            ContentType='application/json'
        )

        logger.info(f"Uploaded results to {target_folder}")

        # 7. Create Metadata Payload & Send to SQS
        metadata_payload = {
            "case_id": case_id,
            "original_document": f"s3://{bucket_name}/{object_key}",
            "processed_markdown": f"s3://{bucket_name}/{md_key}",
            "processed_chunks": f"s3://{bucket_name}/{json_key}",
            "document_metadata": {
                "page_count": doc.page_count,
                "name": doc.name,
                "chunk_count": len(output_chunks)
            },
            "status": "success"
        }

        # Save metadata.json to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=meta_key,
            Body=json.dumps(metadata_payload, indent=2),
            ContentType='application/json'
        )

        # Send to SQS
        if SQS_QUEUE_URL:
            sqs_client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(metadata_payload)
            )
            logger.info("Sent metadata to SQS")
        else:
            logger.warning("SQS_QUEUE_URL not set, skipping queue push")

        return {
            'statusCode': 200,
            'body': json.dumps(metadata_payload)
        }

    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Error: {str(e)}\n{stack_trace}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e), 'trace': stack_trace})
        }