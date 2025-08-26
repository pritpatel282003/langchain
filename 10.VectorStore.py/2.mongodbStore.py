from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
import logging
from typing import List, Dict, Any
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoToChromaVectorStore:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", 
                 db_name: str = "cat_assessment", 
                 collection_name: str = "item_bank",
                 persist_directory: str = "chroma_db_all_questions"):
        """
        Initialize the MongoDB to Chroma vector store converter.
        
        Args:
            mongo_uri: MongoDB connection string
            db_name: MongoDB database name
            collection_name: MongoDB collection name
            persist_directory: Directory to persist Chroma database
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize components
        self.client = None
        self.db = None
        self.collection = None
        self.embedding_model = None
        self.vector_store = None
        
    def connect_to_mongodb(self) -> bool:
        """Connect to MongoDB and return success status."""
        try:
            self.client = MongoClient(self.mongo_uri)
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            logger.info(f"Successfully connected to MongoDB: {self.db_name}.{self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
    
    def fetch_documents_from_mongo(self) -> List[Dict[str, Any]]:
        """Fetch all documents from MongoDB collection."""
        try:
            # You can add filters here if needed
            # e.g., {"isActive": True, "approvedBy": {"$exists": True}}
            query = {"isActive": True}  # Only fetch active questions
            
            mongo_docs = list(self.collection.find(query))
            logger.info(f"Retrieved {len(mongo_docs)} documents from MongoDB")
            return mongo_docs
        except Exception as e:
            logger.error(f"Error fetching documents from MongoDB: {e}")
            return []
    
    def create_page_content(self, doc: Dict[str, Any]) -> str:
        """Create searchable content from MongoDB document."""
        content_parts = []
        
        # Add question title
        if doc.get("questionTitle"):
            content_parts.append(doc["questionTitle"])
        
        # Add explanation
        if doc.get("explanation"):
            content_parts.append(doc["explanation"])
        
        # Add options for better searchability
        if doc.get("options"):
            options_text = []
            for key, value in doc["options"].items():
                options_text.append(f"Option {key}: {value}")
            content_parts.append(" ".join(options_text))
        
        # Add skill description
        if doc.get("skillDescription"):
            content_parts.append(doc["skillDescription"])
        
        # Join all parts with space
        return " ".join(filter(None, content_parts))
    
    def prepare_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and filter metadata for the document."""
        # Essential fields to keep in metadata
        essential_fields = [
            'question_id', 'topic', 'difficultyLevel', 'questionType', 
            'skillCode', 'skillDescription', 'correctOption', 'iseeLevel',
            'a_discrimination', 'b_difficulty', 'run_id'
        ]
        
        # Extract essential metadata with type conversion
        metadata = {}
        for field in essential_fields:
            if field in doc and doc[field] is not None:
                value = doc[field]
                # Convert to simple types that Chroma can handle
                if isinstance(value, (str, int, float, bool)):
                    metadata[field] = value
                else:
                    metadata[field] = str(value)
        
        # Handle special fields separately
        # Options - convert to string representation
        if doc.get('options'):
            try:
                options = doc['options']
                if isinstance(options, dict):
                    metadata['options_text'] = ', '.join([f"{k}: {v}" for k, v in options.items()])
                    metadata['options_count'] = len(options)
                else:
                    metadata['options_text'] = str(options)
            except:
                pass
        
        # Images - just store count and boolean
        if doc.get('questionImages'):
            try:
                images = doc['questionImages']
                if isinstance(images, list):
                    metadata['has_images'] = True
                    metadata['images_count'] = len(images)
                else:
                    metadata['has_images'] = bool(images)
            except:
                metadata['has_images'] = False
        else:
            metadata['has_images'] = False
        
        # Add computed fields
        metadata['document_type'] = 'question'
        
        # Convert timestamps to strings if they exist
        for timestamp_field in ['timestamp', 'createdAt', 'updatedAt', 'approvedAt']:
            if timestamp_field in doc and doc[timestamp_field]:
                try:
                    metadata[f"{timestamp_field}_str"] = str(doc[timestamp_field])
                except:
                    pass
        
        # Manual filtering instead of using filter_complex_metadata
        # Only keep simple types that Chroma can handle
        filtered_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) and value is not None:
                # Additional check for string length (some vector stores have limits)
                if isinstance(value, str) and len(value) > 1000:
                    filtered_metadata[key] = value[:1000] + "..."
                else:
                    filtered_metadata[key] = value
        
        # Ensure we have at least basic metadata
        if not filtered_metadata.get('question_id'):
            filtered_metadata['question_id'] = doc.get('question_id', 'unknown')
        if not filtered_metadata.get('topic'):
            filtered_metadata['topic'] = doc.get('topic', 'unknown')
        if not filtered_metadata.get('difficultyLevel'):
            filtered_metadata['difficultyLevel'] = doc.get('difficultyLevel', 'unknown')
        
        return filtered_metadata
    
    def convert_mongo_docs_to_langchain_docs(self, mongo_docs: List[Dict[str, Any]]) -> List[Document]:
        """Convert MongoDB documents to LangChain Documents."""
        docs = []
        
        for doc in mongo_docs:
            try:
                # Create page content
                page_content = self.create_page_content(doc)
                
                if not page_content.strip():
                    logger.warning(f"Empty content for document {doc.get('question_id', 'unknown')}")
                    continue
                
                # Prepare metadata
                metadata = self.prepare_metadata(doc)
                
                # Use MongoDB _id as document id
                doc_id = str(doc.get("_id", ""))
                
                # Create LangChain Document
                langchain_doc = Document(
                    page_content=page_content,
                    metadata=metadata,
                    id=doc_id
                )
                
                docs.append(langchain_doc)
                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('question_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully converted {len(docs)} documents to LangChain format")
        return docs
    
    def initialize_vector_store(self) -> bool:
        """Initialize the embedding model and Chroma vector store."""
        try:
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # You can change this
                model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have GPU
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize Chroma vector store
            self.vector_store = Chroma(
                collection_name="all_questions",
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
            logger.info("Successfully initialized vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return False
    
    def add_documents_to_vector_store(self, docs: List[Document], batch_size: int = 100) -> bool:
        """Add documents to the vector store in batches."""
        try:
            total_docs = len(docs)
            logger.info(f"Adding {total_docs} documents to vector store...")
            
            # Process in batches to avoid memory issues
            for i in range(0, total_docs, batch_size):
                batch = docs[i:i + batch_size]
                
                try:
                    self.vector_store.add_documents(batch)
                    logger.info(f"Processed batch {i//batch_size + 1}: {min(i + batch_size, total_docs)}/{total_docs} documents")
                except Exception as e:
                    logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                    continue
            
            logger.info("Successfully added all documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return False
    
    def search_similar_documents(self, query: str, k: int = 5, 
                               filter_dict: Dict[str, Any] = None) -> List[tuple]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters (if supported by your Chroma version)
        
        Returns:
            List of (Document, score) tuples
        """
        try:
            if filter_dict:
                # Try with filter (may not be supported in all Chroma versions)
                try:
                    results = self.vector_store.similarity_search_with_score(
                        query=query, k=k, filter=filter_dict
                    )
                except TypeError:
                    # Fallback without filter
                    logger.warning("Filter not supported, searching without filters")
                    results = self.vector_store.similarity_search_with_score(query=query, k=k)
            else:
                results = self.vector_store.similarity_search_with_score(query=query, k=k)
            
            logger.info(f"Found {len(results)} similar documents for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def display_search_results(self, results: List[tuple]):
        """Display search results in a formatted way."""
        if not results:
            print("No results found.")
            return
        
        print(f"\n{'='*80}")
        print(f"SEARCH RESULTS ({len(results)} documents found)")
        print(f"{'='*80}")
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Question ID: {doc.metadata.get('question_id', 'N/A')}")
            print(f"Topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"Difficulty: {doc.metadata.get('difficultyLevel', 'N/A')}")
            print(f"Question Type: {doc.metadata.get('questionType', 'N/A')}")
            print(f"ISEE Level: {doc.metadata.get('iseeLevel', 'N/A')}")
            print(f"Correct Answer: {doc.metadata.get('correctOption', 'N/A')}")
            print(f"Similarity Score: {score:.4f}")
            print(f"Content Preview: {doc.page_content[:200]}...")
            
            if doc.metadata.get('questionImages'):
                print(f"Has Images: Yes ({len(doc.metadata['questionImages'])} image(s))")
            
            print("-" * 50)
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete pipeline from MongoDB to Chroma."""
        logger.info("Starting MongoDB to Chroma vector store pipeline...")
        
        # Step 1: Connect to MongoDB
        if not self.connect_to_mongodb():
            return False
        
        # Step 2: Fetch documents
        mongo_docs = self.fetch_documents_from_mongo()
        if not mongo_docs:
            logger.error("No documents found in MongoDB")
            return False
        
        # Step 3: Convert documents
        langchain_docs = self.convert_mongo_docs_to_langchain_docs(mongo_docs)
        if not langchain_docs:
            logger.error("No valid documents converted")
            return False
        
        # Step 4: Initialize vector store
        if not self.initialize_vector_store():
            return False
        
        # Step 5: Add documents to vector store
        if not self.add_documents_to_vector_store(langchain_docs):
            return False
        
        logger.info("Pipeline completed successfully!")
        return True
    
    def close_connections(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def main():
    """Main function to run the complete pipeline with examples."""
    
    # Initialize the converter
    converter = MongoToChromaVectorStore(
        mongo_uri="mongodb://localhost:27017",
        db_name="cat_assessment", 
        collection_name="item_bank",
        persist_directory="chroma_db_all_questions"
    )
    
    try:
        # Run the complete pipeline
        success = converter.run_complete_pipeline()
        
        if success:
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            # Example searches
            example_queries = [
                "Word problems involving x and y variables",
                "algebra linear equations",
                "quantitative reasoning patterns",
                "middle level mathematics"
            ]
            
            print("\nRunning example searches...")
            
            for query in example_queries:
                print(f"\n🔍 Searching for: '{query}'")
                results = converter.search_similar_documents(query, k=3)
                converter.display_search_results(results)
            
            # Example with metadata filtering (if supported)
            print("\n🔍 Searching with filters...")
            results = converter.search_similar_documents(
                "mathematics word problems", 
                k=5,
                filter_dict={"difficultyLevel": "Easy", "topic": "Word Problems"}
            )
            converter.display_search_results(results)
            
        else:
            print("Pipeline failed. Check the logs for details.")
    
    finally:
        # Clean up
        converter.close_connections()


if __name__ == "__main__":
    main()