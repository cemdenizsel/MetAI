# RAGFlow Internal RAG Engine Setup

This repository contains the complete setup for RAGFlow as an internal RAG engine for document ingestion, parsing, chunking, embedding, indexing, and retrieval. The system is designed to support "live document lookup" functionality for meeting analysis applications.

## Architecture Overview

The setup includes:
- RAGFlow application with API and web interfaces
- MySQL for metadata storage
- Redis for caching and queues
- Milvus for vector storage and similarity search
- MinIO for object storage
- Etcd for coordination

## Components

### Core Infrastructure (`docker-compose.ragflow.yml`)
- Complete container orchestration for all required services
- Persistent volumes for data durability
- Proper networking configuration

### Configuration (`.env.ragflow.example`)
- Environment variable templates for easy customization
- Secure default settings
- Performance tuning parameters

### Documentation
- `RAGFLOW_OPERATIONS_GUIDE.md`: Operational procedures for dataset management
- `PDF_HIGHLIGHT_METADATA.md`: Configuration for PDF metadata extraction
- `MULTILINGUAL_EMBEDDING_CONFIG.md`: Embedding and chunking recommendations
- `PERFORMANCE_GUIDELINES.md`: Performance optimization strategies
- `VALIDATION_CHECKLIST.md`: Deployment verification steps

### Test Suite (`test/`)
- Integration tests with dummy documents
- User isolation validation
- Metadata verification
- Retrieval functionality testing

## Deployment

### Prerequisites
- Docker and Docker Compose
- At least 8GB RAM and 4 CPU cores
- 50GB+ free disk space

### Quick Start
1. Copy `.env.ragflow.example` to `.env` and customize values
2. Run: `docker-compose -f docker-compose.ragflow.yml up -d`
3. Follow the validation checklist in `VALIDATION_CHECKLIST.md`

## Key Features

### Per-User Data Isolation
- Dataset naming convention: `kb_user_<user_id>`
- Strict data separation between users
- Efficient querying within user boundaries

### PDF Highlight Support
- Page number extraction for each chunk
- Bounding box information when available
- Position metadata for precise highlighting

### Multilingual Support
- BGE-M3 embedding model supporting 100+ languages
- Optimized for German/English/Turkish
- Cross-lingual semantic understanding

### Performance Optimized
- Configurable chunking strategies
- Vector index optimization
- Caching mechanisms
- Resource utilization controls

## Testing

The test suite in the `test/` directory validates:
- Document ingestion and processing
- Metadata extraction completeness
- User data isolation
- Retrieval functionality
- Multilingual support

Run the tests with:
```bash
cd test
pip install -r requirements.txt
python test_ragflow_integration.py
```

## Maintenance

Regular maintenance tasks include:
- Monitoring service health
- Managing storage volumes
- Updating configurations as needed
- Performing backups of critical data