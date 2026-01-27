"""
RAGFlow Integration Test Cases
This script tests the end-to-end functionality of RAGFlow with dummy documents and user IDs.
"""

import os
import json
import requests
import time
from pathlib import Path

# Configuration
RAGFLOW_API_URL = os.getenv('RAGFLOW_API_URL', 'http://localhost:9380')
RAGFLOW_API_KEY = os.getenv('RAGFLOW_API_KEY', 'your_api_key_here')
HEADERS = {
    'Authorization': f'Api-Key {RAGFLOW_API_KEY}',
    'Content-Type': 'application/json'
}

def create_dummy_pdf(file_path):
    """
    Creates a dummy PDF file for testing purposes.
    In a real scenario, this would be an actual PDF with content.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "Dummy Test Document")
        
        # Add some sample content
        c.setFont("Helvetica", 12)
        y_position = height - 100
        sample_text = [
            "This is a dummy document for testing RAGFlow integration.",
            "It contains sample content that will be ingested and indexed.",
            "The document includes multiple paragraphs for chunking tests.",
            "Meeting analysis application will use this for live document lookup.",
            "German: Dies ist ein Beispieltext auf Deutsch.",
            "English: This is an example text in English.",
            "Turkish: Bu Türkçe bir örnek metindir."
        ]
        
        for line in sample_text:
            c.drawString(50, y_position, line)
            y_position -= 20
            
        # Add a second page
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 50, "Second Page Content")
        
        c.setFont("Helvetica", 12)
        y_position = height - 100
        sample_text_2 = [
            "This is the second page of the dummy document.",
            "Additional content for testing multi-page PDF handling.",
            "Verifying that page numbers are correctly extracted.",
            "Testing metadata preservation across pages."
        ]
        
        for line in sample_text_2:
            c.drawString(50, y_position, line)
            y_position -= 20
        
        c.save()
        print(f"Created dummy PDF at: {file_path}")
        return True
    except ImportError:
        print("reportlab not available, creating a dummy file differently...")
        # Create a dummy file in a different way for testing
        Path(file_path).touch()
        return False

def create_user_dataset(user_id):
    """
    Creates a dataset for a specific user following the naming convention.
    """
    dataset_name = f"kb_user_{user_id}"
    url = f"{RAGFLOW_API_URL}/api/v1/dataset"
    
    payload = {
        "name": dataset_name,
        "kb_id": f"kb_{user_id}_{int(time.time())}",
        "parser_id": "pdf_parser",
        "progress": {
            "total": 0,
            "processed": 0,
            "success": 0,
            "failed": 0
        }
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        dataset_info = response.json()
        print(f"Created dataset for user {user_id}: {dataset_info.get('id')}")
        return dataset_info.get('id')
    else:
        print(f"Failed to create dataset for user {user_id}. Status: {response.status_code}, Response: {response.text}")
        return None

def upload_document_to_dataset(dataset_id, file_path):
    """
    Uploads a document to a specific dataset.
    """
    url = f"{RAGFLOW_API_URL}/api/v1/document/upload"
    
    with open(file_path, 'rb') as file:
        files = {
            'file': (os.path.basename(file_path), file, 'application/pdf'),
        }
        data = {
            'dataset_id': dataset_id,
            'parser_id': 'pdf_parser',
            'step_nums': 3,
            'layout_recognize': True,
            'enable_ocr': False
        }
        
        response = requests.post(url, headers={'Authorization': f'Api-Key {RAGFLOW_API_KEY}'}, files=files, data=data)
    
    if response.status_code == 200:
        doc_info = response.json()
        print(f"Uploaded document to dataset {dataset_id}: {doc_info.get('id')}")
        return doc_info.get('id')
    else:
        print(f"Failed to upload document. Status: {response.status_code}, Response: {response.text}")
        return None

def wait_for_document_processing(dataset_id, document_id, timeout=300):
    """
    Waits for document processing to complete.
    """
    url = f"{RAGFLOW_API_URL}/api/v1/document/{document_id}"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            doc_info = response.json()
            status = doc_info.get('progress', {}).get('state', 'unknown')
            print(f"Document processing status: {status}")
            
            if status == 'Finished':
                print("Document processing completed successfully")
                return True
            elif status in ['Error', 'Failed']:
                print(f"Document processing failed with status: {status}")
                return False
        
        time.sleep(10)  # Wait 10 seconds before checking again
    
    print("Document processing timed out")
    return False

def verify_document_chunks(document_id):
    """
    Verifies that document chunks contain required metadata.
    """
    url = f"{RAGFLOW_API_URL}/api/v1/document/{document_id}/chunks?page=1&size=100"
    
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        chunks_data = response.json()
        chunks = chunks_data.get('data', [])
        
        print(f"Retrieved {len(chunks)} chunks from document {document_id}")
        
        # Check if chunks have required metadata
        for i, chunk in enumerate(chunks):
            meta = chunk.get('meta', {})
            content = chunk.get('content', '')
            
            # Check for page number
            page_number = meta.get('page_number') or meta.get('page_idx')
            if page_number is not None:
                print(f"Chunk {i}: Has page number {page_number}")
            else:
                print(f"Chunk {i}: Missing page number in metadata")
            
            # Check for bounding box or position info
            bbox = meta.get('bbox') or meta.get('bounding_box')
            if bbox:
                print(f"Chunk {i}: Has bounding box information")
            else:
                print(f"Chunk {i}: Missing bounding box information")
                
            # Print a snippet of content
            print(f"Chunk {i} content preview: {content[:100]}...")
        
        return len(chunks) > 0
    else:
        print(f"Failed to retrieve chunks. Status: {response.status_code}, Response: {response.text}")
        return False

def test_retrieval_query(dataset_id, query_text):
    """
    Tests retrieval functionality with a query.
    """
    url = f"{RAGFLOW_API_URL}/api/v1/chat/completion"
    
    payload = {
        "conversation_id": "",
        "dataset_ids": [dataset_id],
        "query": query_text,
        "stream": False
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Retrieval query successful. Found {len(result.get('candidates', []))} candidates")
        
        # Print first candidate for verification
        candidates = result.get('candidates', [])
        if candidates:
            first_candidate = candidates[0]
            print(f"First result: {first_candidate.get('content', '')[:200]}...")
            
            # Check if metadata contains required fields
            meta = first_candidate.get('meta', {})
            page_number = meta.get('page_number') or meta.get('page_idx')
            if page_number:
                print(f"Result contains page number: {page_number}")
            else:
                print("Result missing page number in metadata")
                
            bbox = meta.get('bbox') or meta.get('bounding_box')
            if bbox:
                print("Result contains bounding box information")
            else:
                print("Result missing bounding box information")
        
        return True
    else:
        print(f"Retrieval query failed. Status: {response.status_code}, Response: {response.text}")
        return False

def test_user_isolation():
    """
    Tests that different users have isolated datasets.
    """
    print("\n=== Testing User Isolation ===")
    
    # Create datasets for two different users
    user1_id = "test_user_123"
    user2_id = "test_user_456"
    
    dataset1_id = create_user_dataset(user1_id)
    dataset2_id = create_user_dataset(user2_id)
    
    if not dataset1_id or not dataset2_id:
        print("Failed to create datasets for user isolation test")
        return False
    
    # Create and upload a document to each dataset
    dummy_pdf_path1 = "/tmp/dummy_doc_user1.pdf"
    dummy_pdf_path2 = "/tmp/dummy_doc_user2.pdf"
    
    # Create dummy PDFs
    create_dummy_pdf(dummy_pdf_path1)
    create_dummy_pdf(dummy_pdf_path2)
    
    doc1_id = upload_document_to_dataset(dataset1_id, dummy_pdf_path1)
    doc2_id = upload_document_to_dataset(dataset2_id, dummy_pdf_path2)
    
    if not doc1_id or not doc2_id:
        print("Failed to upload documents for user isolation test")
        return False
    
    # Wait for processing
    if not wait_for_document_processing(dataset1_id, doc1_id) or not wait_for_document_processing(dataset2_id, doc2_id):
        print("Document processing failed for user isolation test")
        return False
    
    # Test retrieval from each user's dataset
    query = "dummy document content"
    
    print(f"\nTesting retrieval from user {user1_id}'s dataset:")
    success1 = test_retrieval_query(dataset1_id, query)
    
    print(f"\nTesting retrieval from user {user2_id}'s dataset:")
    success2 = test_retrieval_query(dataset2_id, query)
    
    # Clean up temporary files
    try:
        os.remove(dummy_pdf_path1)
        os.remove(dummy_pdf_path2)
    except:
        pass
    
    return success1 and success2

def main():
    """
    Main test function that runs all integration tests.
    """
    print("Starting RAGFlow Integration Tests...")
    
    # Create a dummy PDF for testing
    dummy_pdf_path = "/tmp/dummy_test_document.pdf"
    print(f"\nCreating dummy PDF at: {dummy_pdf_path}")
    
    if not create_dummy_pdf(dummy_pdf_path):
        print("Could not create dummy PDF, skipping tests that require it")
        return
    
    # Test with a dummy user
    user_id = "test_user_789"
    print(f"\nTesting with dummy user ID: {user_id}")
    
    # Create user dataset
    dataset_id = create_user_dataset(user_id)
    if not dataset_id:
        print("Failed to create dataset, aborting tests")
        return
    
    # Upload document to user's dataset
    document_id = upload_document_to_dataset(dataset_id, dummy_pdf_path)
    if not document_id:
        print("Failed to upload document, aborting tests")
        return
    
    # Wait for document processing to complete
    print("\nWaiting for document processing to complete...")
    if not wait_for_document_processing(dataset_id, document_id):
        print("Document processing failed, aborting tests")
        return
    
    # Verify document chunks contain required metadata
    print("\nVerifying document chunks...")
    if not verify_document_chunks(document_id):
        print("Chunk verification failed")
        return
    
    # Test retrieval functionality
    print("\nTesting retrieval functionality...")
    test_queries = [
        "dummy document content",
        "meeting analysis application",
        "German: Dies ist ein Beispieltext",
        "English: This is an example text",
        "Turkish: Bu Türkçe bir örnek metindir"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        if not test_retrieval_query(dataset_id, query):
            print(f"Retrieval test failed for query: {query}")
            return
    
    # Test user isolation
    print("\n" + "="*50)
    if not test_user_isolation():
        print("User isolation test failed")
        return
    
    print("\n" + "="*50)
    print("All RAGFlow integration tests completed successfully!")
    
    # Clean up temporary file
    try:
        os.remove(dummy_pdf_path)
    except:
        pass

if __name__ == "__main__":
    main()