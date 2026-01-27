#!/usr/bin/env python3
"""
Dummy PDF Generator for RAGFlow Testing
This script creates a sample PDF document that can be used for testing the RAGFlow setup.
"""

import os
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def create_dummy_pdf(output_path):
    """
    Creates a dummy PDF with sample content for testing RAGFlow.
    """
    # Register a standard font
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Title page
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 100, "Meeting Analysis Reference Document")
    
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height - 130, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add some sample content
    y_position = height - 180
    sample_sections = [
        ("Executive Summary", [
            "This document serves as a reference for meeting analysis and document lookup functionality.",
            "It contains sample content that will be processed by the RAGFlow system.",
            "Various types of content are included to test different processing capabilities."
        ]),
        ("Project Updates", [
            "Current project status includes several milestones achieved this quarter.",
            "Team A has completed phase 1 of the development cycle.",
            "Budget allocation has been adjusted based on preliminary results."
        ]),
        ("Technical Specifications", [
            "System requirements include minimum 8GB RAM and 4 CPU cores.",
            "Supported file formats: PDF, DOCX, TXT, PPTX, XLSX.",
            "Maximum file size: 100MB per document."
        ]),
        ("Multilingual Support", [
            "English: This section demonstrates English content processing.",
            "German: Dieser Abschnitt zeigt die Verarbeitung von deutschsprachigen Inhalten.",
            "Turkish: Bu bölüm Türkçe içerik işleme özelliğini göstermektedir."
        ])
    ]
    
    for section_title, content_list in sample_sections:
        # Section title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, section_title)
        y_position -= 30
        
        # Section content
        c.setFont("Helvetica", 10)
        for line in content_list:
            # Split long lines to fit page width
            max_line_width = 500
            if len(line) > 80:  # Rough estimate for line length
                # Simple word wrap
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) < 80:
                        current_line += " " + word if current_line else word
                    else:
                        c.drawString(70, y_position, current_line)
                        y_position -= 15
                        current_line = word
                if current_line:
                    c.drawString(70, y_position, current_line)
                    y_position -= 15
            else:
                c.drawString(70, y_position, line)
                y_position -= 15
            
            # Add extra space between paragraphs
            y_position -= 5
            
            # Check if we need a new page
            if y_position < 100:
                c.showPage()
                y_position = height - 50
    
    # Add a second page with more content
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, height - 50, "Additional Content - Page 2")
    
    y_position = height - 100
    c.setFont("Helvetica", 10)
    additional_content = [
        "This is the second page of the dummy document.",
        "It contains additional content for testing multi-page PDF processing.",
        "The RAGFlow system should correctly identify page boundaries.",
        "Metadata extraction should include page numbers for each text segment.",
        "Different content types are included to test various parsing capabilities.",
        "Tables, lists, and formatted text may be added in future versions.",
        "This concludes the sample content for testing purposes.",
        "Meeting analysis features will use this document for live lookup tests."
    ]
    
    for line in additional_content:
        c.drawString(50, y_position, line)
        y_position -= 15
        if y_position < 100:
            c.showPage()
            y_position = height - 50
    
    # Finalize the PDF
    c.save()
    print(f"Dummy PDF created successfully at: {output_path}")

def main():
    output_path = "/tmp/dummy_meeting_document.pdf"
    create_dummy_pdf(output_path)
    print("You can now use this dummy PDF for RAGFlow testing.")

if __name__ == "__main__":
    main()