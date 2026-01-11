
import pytest
import pandas as pd
import numpy as np
import hashlib

def test_hashing_collision_rate_simulation():
    # Create high cardinality data
    n_students = 10000
    student_ids = [f"STU_{i}" for i in range(n_students)]
    
    bucket_size = 100000 # Increased to ensure low collision
    
    hashes = [int(hashlib.md5(sid.encode()).hexdigest(), 16) % bucket_size for sid in student_ids]
    unique_hashes = len(set(hashes))
    collisions = n_students - unique_hashes
    collision_rate = collisions / n_students
    
    # Assert collision rate is acceptable for a hashing approach (< 15%)
    # This justifies why we might want to stay with Embedding or ensure bucket is large enough
    assert collision_rate < 0.15, f"High collision rate: {collision_rate:.2%}"


