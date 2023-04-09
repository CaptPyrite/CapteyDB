import Vector

for face_index, faces in enumerate(Vector.os.listdir("Key-faces")):
    
    vtc = Vector.generate_key_vectors_from_image(f"Key-faces/{faces}", f"{face_index}")

print(vtc.vectors)