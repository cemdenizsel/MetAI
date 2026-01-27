# RAGFlow Test Suite

This directory contains integration tests for the RAGFlow setup, specifically designed to test document ingestion, parsing, and retrieval with user isolation.

## Test Components

### 1. Dummy Document Creation
- Creates a sample PDF document with multilingual content (German/English/Turkish)
- Includes multiple pages to test page number extraction
- Contains various content types for chunking validation

### 2. User Isolation Testing
- Creates datasets following the naming convention `kb_user_<user_id>`
- Tests that different users have isolated document collections
- Verifies that queries to one user's dataset don't return results from another's

### 3. Metadata Validation
- Verifies that document chunks contain page numbers
- Checks for bounding box information needed for PDF highlighting
- Ensures content is properly preserved during ingestion

### 4. Retrieval Testing
- Performs similarity searches with various query types
- Validates that results contain necessary metadata for highlighting
- Tests multilingual query support

## Prerequisites

Before running the tests, ensure:

1. RAGFlow is running and accessible
2. You have a valid API key
3. The following environment variables are set:
   - `RAGFLOW_API_URL` (default: http://localhost:9380)
   - `RAGFLOW_API_KEY` (your RAGFlow API key)

## Installation

Install the required dependencies:

```bash
pip install requests reportlab
```

## Running Tests

Execute the test suite:

```bash
cd ragflow/test
python test_ragflow_integration.py
```

## Test Scenarios

The test suite performs the following scenarios:

1. **Document Ingestion Test**:
   - Creates a dummy PDF document
   - Uploads it to a user-specific dataset
   - Waits for processing to complete
   - Verifies chunk metadata

2. **Retrieval Test**:
   - Performs multiple queries against the ingested document
   - Verifies results contain proper metadata
   - Tests multilingual query support

3. **User Isolation Test**:
   - Creates datasets for multiple users
   - Uploads different documents to each
   - Verifies that queries remain isolated by user

## Expected Results

Upon successful completion, the tests should show:
- Document ingestion completed successfully
- All chunks contain page numbers and metadata
- Retrieval queries return relevant results with proper metadata
- User isolation is maintained across datasets