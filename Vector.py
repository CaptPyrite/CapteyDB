import os
import random
import json
import facial_landmarks

class Vectorization:
    vectors = {}

vectors = Vectorization.vectors

def generate_key_vectors_from_image(image:os.path, vector_title:str) -> Vectorization:
    if len(Vectorization.vectors) <= 1:
        Vectorization.vectors[vector_title] = [random.randint(0, len(vectors)), # X
                                               random.randint(0, len(vectors)), # Y
                                               random.randint(0, len(vectors))] # Z    
    else:
        facial_landmarks.compare_between_3_faces()
        pass
        
    return Vectorization

def save(file:os.path) -> Vectorization:
    json.dump(Vectorization.vectors, open(file,"w"), indent=2)
