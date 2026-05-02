import numpy as np
import cv2
import pytest
from PIL import Image
import os
from clean_pdf import clean_page

def test_clean_page_shapes():
    # Create a dummy yellowish image (BGR)
    # Yellowish is high Red, high Green, low Blue
    img = np.full((100, 100, 3), (200, 240, 240), dtype=np.uint8) # BGR: light yellow-ish
    # Add a black "text" block
    img[40:60, 40:60] = [0, 0, 0]
    
    cleaned = clean_page(img, level=3, deskew=False)
    
    assert cleaned.shape == (100, 100, 3)
    # Check if the yellowish background turned white [255, 255, 255]
    # The center should still be dark
    assert np.all(cleaned[0, 0] == [255, 255, 255])
    assert np.all(cleaned[50, 50] == [0, 0, 0])

def test_clean_page_levels():
    # Test that different levels don't crash
    img = np.random.randint(200, 255, (100, 100, 3), dtype=np.uint8)
    for level in range(1, 6):
        cleaned = clean_page(img, level=level)
        assert cleaned.shape == (100, 100, 3)

def test_deskew_logic_no_crash():
    # Create an image with some "text" at an angle
    img = np.full((200, 200, 3), (255, 255, 255), dtype=np.uint8)
    # Draw a tilted rectangle (simulating text lines)
    pts = np.array([[50, 50], [150, 60], [145, 80], [45, 70]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 0))
    
    cleaned = clean_page(img, level=3, deskew=True)
    assert cleaned.shape == (200, 200, 3)
